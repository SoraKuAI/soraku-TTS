import math
import numpy as np
from scipy.signal import resample_poly

def rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms <= 1e-10:
        return -120.0
    return 20.0 * math.log10(rms)

def peak_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-12:
        return -120.0
    return 20.0 * math.log10(peak)

def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return np.mean(y, axis=1, dtype=np.float32)
    raise ValueError(f"Unexpected audio shape: {y.shape}")

def resample_audio(y: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return y.astype(np.float32, copy=False)
    g = math.gcd(int(src_sr), int(target_sr))
    up = target_sr // g
    down = src_sr // g
    out = resample_poly(y, up, down)
    return np.asarray(out, dtype=np.float32)

def _frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """分块向量化计算每帧 RMS，避免逐帧 Python 循环拖慢多核扩展。"""
    n_frames = 1 + (len(y) - frame) // hop
    rms = np.empty(n_frames, dtype=np.float32)
    eps = 1e-8

    # 每次最多计算 2048 帧，避免 sliding_window_view 产生过大的中间数组。
    chunk_frames = 2048
    for start_frame in range(0, n_frames, chunk_frames):
        end_frame = min(n_frames, start_frame + chunk_frames)
        start_sample = start_frame * hop
        end_sample = start_sample + (end_frame - start_frame - 1) * hop + frame

        view = np.lib.stride_tricks.sliding_window_view(
            y[start_sample:end_sample], frame, axis=0
        )[::hop]
        rms[start_frame:end_frame] = np.sqrt(
            np.mean(view * view, axis=1) + eps
        )

    return rms

def trim_silence(
    y: np.ndarray,
    sr: int,
    top_db: float,
    max_tail_silence: float,
) -> np.ndarray:
    """粗粒度 RMS 活动区裁剪：去头静音，尾部最多保留 max_tail_silence。"""
    if y.size == 0:
        return y

    frame = max(1, int(sr * 0.03))
    hop = max(1, int(sr * 0.01))
    if len(y) < frame:
        return y

    rms = _frame_rms(y, frame, hop)
    eps = 1e-8

    db = 20.0 * np.log10(rms + eps)
    threshold = float(np.max(db)) - top_db
    active = db >= threshold

    idx = np.flatnonzero(active)
    if idx.size == 0:
        return y[:0]

    content_start = int(idx[0]) * hop
    content_end = min(len(y), int(idx[-1]) * hop + frame)
    tail = int(sr * max_tail_silence)
    trim_end = min(len(y), content_end + tail)

    return y[content_start:trim_end]

def normalize_rms(y: np.ndarray, target_db: float) -> np.ndarray:
    current = rms_db(y)
    if current <= -100.0:
        return y

    gain_db = target_db - current
    out = y * (10.0 ** (gain_db / 20.0))

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return np.asarray(out, dtype=np.float32)
