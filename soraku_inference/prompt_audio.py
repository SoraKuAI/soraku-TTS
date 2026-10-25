"""推理侧 prompt 音频预处理：与训练数据管线同一套口径。

训练数据（prepare_pre_train_datasets.process_audio_item）的顺序是：
单声道 → 重采样 24 kHz → 静音裁剪（去头、尾部最多 0.5s）→ RMS 归一化到 -23 dB。
本模块把同一套函数复用到 zero-shot 推理的外部参考音频上，避免高能量或
响度口径不一致的 prompt 把模型带偏。不做降噪/混响处理（训练默认同样不做）。
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio_ops import (
    normalize_rms,
    peak_db,
    resample_audio,
    rms_db,
    to_mono,
    trim_silence,
)

CODEC_INPUT_SR = 24000
CODEC_FRAME_RATE = 12.5


@dataclass(frozen=True)
class PromptAudioResult:
    path: Path
    original_duration: float
    original_rms_db: float
    original_sample_rate: int
    clean_duration: float
    clean_rms_db: float
    clean_peak_db: float
    frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "original_duration": round(self.original_duration, 3),
            "original_rms_db": round(self.original_rms_db, 1),
            "original_sample_rate": self.original_sample_rate,
            "clean_duration": round(self.clean_duration, 3),
            "clean_rms_db": round(self.clean_rms_db, 1),
            "clean_peak_db": round(self.clean_peak_db, 1),
            "frames": self.frames,
        }


def prepare_prompt_audio(
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    target_sr: int = CODEC_INPUT_SR,
    target_db: float = -23.0,
    silence_top_db: float = 32.0,
    max_tail_silence: float = 0.5,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> PromptAudioResult:
    """把外部 prompt 音频处理成训练口径的 24kHz 单声道干净 WAV。

    输出按内容 sha1 命名落在 output_dir，重复上传同一音频直接复用。
    """
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"prompt audio not found: {source}")
    with sf.SoundFile(source) as audio_file:
        src_sr = int(audio_file.samplerate)
        original_channels = int(audio_file.channels)
        original_duration = float(audio_file.frames) / src_sr if src_sr > 0 else 0.0
        y = audio_file.read(dtype="float32", always_2d=False)
    y = to_mono(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        raise ValueError("prompt audio is empty")
    original_rms = rms_db(y)
    y = resample_audio(y, src_sr, target_sr)
    y = trim_silence(y, target_sr, silence_top_db, max_tail_silence)
    clean_duration = float(len(y) / target_sr)
    if clean_duration < min_duration:
        raise ValueError(f"prompt 裁剪后过短：{clean_duration:.2f}s < {min_duration}s")
    if clean_duration > max_duration:
        raise ValueError(
            f"prompt 裁剪后过长：{clean_duration:.2f}s > {max_duration}s；训练时 prompt 上限为 144 帧（11.5s），请截取片段"
        )
    y = normalize_rms(y, target_db)
    digest = hashlib.sha1(
        (str(source.resolve()) + f"|{original_duration}|{target_db}").encode("utf-8")
    ).hexdigest()[:16]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{digest}_{source.stem}.wav"
    sf.write(str(out_path), y, target_sr, subtype="PCM_16")
    return PromptAudioResult(
        path=out_path,
        original_duration=original_duration,
        original_rms_db=original_rms,
        original_sample_rate=src_sr,
        clean_duration=clean_duration,
        clean_rms_db=rms_db(y),
        clean_peak_db=peak_db(y),
        frames=int(round(clean_duration * CODEC_FRAME_RATE)),
    )
