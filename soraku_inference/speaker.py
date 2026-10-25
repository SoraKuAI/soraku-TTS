"""Speaker embedding 提取库。

作为库使用:
    from speaker_embedding import SpeakerEmbeddingExtractor

    with SpeakerEmbeddingExtractor("model.onnx", num_thread=8) as extractor:
        emb = extractor.extract_from_file("a.wav")
        utt2emb = extractor.extract_batch({"utt1": "a.wav", "utt2": "b.wav"})

作为 CLI 使用（与原版行为一致）:
    python speaker_embedding.py --dir data/ --onnx_path model.onnx --num_thread 8
"""

import argparse
import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import onnxruntime
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

__all__ = ["SpeakerEmbeddingExtractor", "read_kaldi_scp"]

PathLike = Union[str, os.PathLike]


def read_kaldi_scp(path: PathLike) -> Dict[str, str]:
    """读取 Kaldi 风格的两列文件（wav.scp / utt2spk 等）。"""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def _load_audio(wav_spec: str) -> Tuple[torch.Tensor, int]:
    """支持普通路径和 Kaldi pipe 风格（"sox xxx - |"）的 wav.scp 条目。"""
    if wav_spec.endswith("|"):
        raw = subprocess.run(
            wav_spec[:-1], shell=True, stdout=subprocess.PIPE, check=True
        ).stdout
        source = io.BytesIO(raw)
    else:
        source = wav_spec
    # soundfile 避免解码依赖 TorchCodec/FFmpeg 动态库；保持 float32 和 [C,T] 接口。
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), int(sample_rate)


class SpeakerEmbeddingExtractor:
    """基于 ONNX 模型的 speaker embedding 提取器。

    注意：onnxruntime 的 ``InferenceSession.run`` 是线程安全的，
    因此内部用 ThreadPoolExecutor 共享同一个 session 做并发提取。
    """

    def __init__(
        self,
        onnx_path: PathLike,
        num_thread: int = 8,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        providers: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.show_progress = show_progress

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # 由外层 ThreadPoolExecutor 提供并发；每个 session.run 固定单线程，
        # 避免 ORT 内部线程池在 64C/128T 机器上和外层线程互相抢占。
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=list(providers)
        )
        self.input_name = self.session.get_inputs()[0].name

        self.executor = ThreadPoolExecutor(max_workers=num_thread)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "SpeakerEmbeddingExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def compute_fbank(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            dither=0,
            sample_frequency=self.sample_rate,
        )
        return feat - feat.mean(dim=0, keepdim=True)

    def extract_from_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[float]:
        """从音频 tensor 提取 embedding。"""
        feat = self.compute_fbank(waveform, sample_rate)
        output = self.session.run(
            None, {self.input_name: feat.unsqueeze(0).cpu().numpy()}
        )[0]
        return output.flatten().tolist()

    def extract_from_file(self, wav_path: PathLike) -> List[float]:
        """从音频文件提取 embedding。"""
        waveform, sample_rate = _load_audio(str(wav_path))
        return self.extract_from_waveform(waveform, sample_rate)

    # 批量提取

    def extract_batch(self, utt2wav: Mapping[str, PathLike]) -> Dict[str, List[float]]:
        """多线程批量提取，返回 utt -> embedding。"""
        future2utt = {
            self.executor.submit(self.extract_from_file, wav): utt
            for utt, wav in utt2wav.items()
        }
        utt2embedding = {}
        futures = as_completed(future2utt)
        if self.show_progress:
            futures = tqdm(futures, total=len(future2utt), desc="extracting")
        for future in futures:
            utt = future2utt[future]
            utt2embedding[utt] = future.result()
        return utt2embedding

    @staticmethod
    def average_to_speaker(
        utt2embedding: Mapping[str, Sequence[float]],
        utt2spk: Mapping[str, str],
    ) -> Dict[str, List[float]]:
        """把 utt 级 embedding 按 speaker 求平均。"""
        spk2embs: Dict[str, List[Sequence[float]]] = {}
        for utt, emb in utt2embedding.items():
            spk2embs.setdefault(utt2spk[utt], []).append(emb)
        return {
            spk: torch.tensor(embs, dtype=torch.float32).mean(dim=0).tolist()
            for spk, embs in spk2embs.items()
        }

    def extract_kaldi_dir(
        self, data_dir: PathLike, save: bool = False
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """读取目录下的 wav.scp / utt2spk，返回 (utt2embedding, spk2embedding)。

        save=True 时同时把结果保存为 utt2embedding.pt / spk2embedding.pt。
        """
        data_dir = str(data_dir)
        utt2wav = read_kaldi_scp(os.path.join(data_dir, "wav.scp"))
        utt2spk = read_kaldi_scp(os.path.join(data_dir, "utt2spk"))
        utt2embedding = self.extract_batch(utt2wav)
        spk2embedding = self.average_to_speaker(utt2embedding, utt2spk)
        if save:
            torch.save(utt2embedding, os.path.join(data_dir, "utt2embedding.pt"))
            torch.save(spk2embedding, os.path.join(data_dir, "spk2embedding.pt"))
        return utt2embedding, spk2embedding


# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 utt/spk 级 speaker embedding")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Kaldi data 目录（需包含 wav.scp 和 utt2spk）",
    )
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--num_thread", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SpeakerEmbeddingExtractor(
        args.onnx_path, num_thread=args.num_thread
    ) as extractor:
        extractor.extract_kaldi_dir(args.dir, save=True)


if __name__ == "__main__":
    main()
