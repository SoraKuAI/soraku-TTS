"""Decode Qwen3-TTS 12 Hz codes to waveforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class CodecDecoder:
    """Load the Qwen codec once and reuse it across an evaluation run."""

    def __init__(self, tokenizer_path: str | Path, *, device: str = "cuda:0") -> None:
        from qwen_tts import Qwen3TTSTokenizer

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(tokenizer_path), device_map=device
        )

    def decode_to_file(
        self, codes: torch.Tensor, output_path: str | Path
    ) -> tuple[Path, float]:
        """Write a WAV and return its path plus duration in seconds."""

        import soundfile as sf

        waveform, sample_rate = self.decode(codes)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, int(sample_rate))
        return output_path, float(len(waveform)) / float(sample_rate)

    def encode_file(self, audio_path: str | Path) -> torch.LongTensor:
        """Encode one audio file into ``[T, 16]`` acoustic codes."""

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"reference audio not found: {audio_path}")
        encoded = self.tokenizer.encode(str(audio_path))
        codes = encoded.audio_codes
        if isinstance(codes, (list, tuple)):
            if len(codes) != 1:
                raise RuntimeError(
                    f"expected one encoded reference, received {len(codes)}"
                )
            codes = codes[0]
        return _validate(codes)

    def decode(self, codes: torch.Tensor) -> tuple[np.ndarray, int]:
        codes = _validate(codes)
        waveforms, sample_rate = self.tokenizer.decode({"audio_codes": codes})
        if isinstance(waveforms, torch.Tensor):
            waveforms = waveforms.detach().cpu().numpy()
        waveforms = np.asarray(waveforms)
        waveform = waveforms[0] if waveforms.ndim == 2 else waveforms
        if waveform.ndim != 1:
            raise RuntimeError(f"unexpected decoded waveform shape: {waveform.shape}")
        return waveform, int(sample_rate)


def _validate(codes: torch.Tensor) -> torch.LongTensor:
    codes = torch.as_tensor(codes, dtype=torch.long).cpu()
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise ValueError(f"codec codes must be [T,16], got {tuple(codes.shape)}")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= 2048):
        raise ValueError("codec decoder received a non-acoustic token")
    return codes
