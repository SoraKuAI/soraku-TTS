"""Portable Soraku inference: local safetensors + tokenizer + architecture config."""

from pathlib import Path
import json

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model import SorakuTTSv2, SorakuV2Config
from .languages import configure_language_tokens, prefix_language


def load_model(model_dir, device="cuda:0"):
    directory = Path(model_dir)
    metadata = json.loads((directory / "config.json").read_text())
    if (
        metadata.get("format_version") != 1
        or metadata.get("model_type") != "soraku_tts_v2"
    ):
        raise ValueError("not a supported Soraku release model directory")
    parameters = dict(metadata["soraku_config"])
    parameters["text_model_path"] = str(directory)
    parameters["language_token_ids"] = tuple(parameters.get("language_token_ids", ()))
    config = SorakuV2Config(**parameters)
    text_config = AutoConfig.for_model(
        **json.loads((directory / "text_config.json").read_text())
    )
    # No base-model weights are loaded: all trunk tensors are in model.safetensors.
    # Construct on CPU so non-persistent RoPE buffers are initialized as well.
    # They are deliberately absent from the state dict and cannot stay on meta.
    trunk = AutoModel.from_config(
        text_config, dtype=torch.float32, attn_implementation="sdpa"
    )
    model = SorakuTTSv2(config, trunk=trunk)
    state = load_file(directory / "model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        directory / "tokenizer", local_files_only=True
    )
    if config.language_token_ids:
        configure_language_tokens(tokenizer, config.language_token_ids)
    return model, tokenizer, metadata


@torch.inference_mode()
def synthesize(
    *,
    model_dir,
    codec_dir,
    reference_audio,
    reference_text,
    text,
    reference_language,
    language,
    output,
    speaker_encoder=None,
    device="cuda:0",
    seed=42,
    cfg_scale=1.5,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    max_new_frames=400,
    backend="native",
):
    import numpy as np
    from .codec import CodecDecoder
    from .prompt_audio import prepare_prompt_audio

    if not reference_text.strip() or not text.strip():
        raise ValueError(
            "verbatim reference transcript and nonempty target text are required"
        )
    model, tokenizer, metadata = load_model(model_dir, device)
    config = model.config
    if not config.language_token_ids and language != "zh":
        raise ValueError(
            "the 102k Chinese model requires language=zh; it was not trained for English/Japanese"
        )
    if config.speaker_conditioning and backend != "native":
        raise ValueError(
            "speaker-conditioned models require native; fast/graph speaker prefill is not validated"
        )
    if config.speaker_conditioning and not speaker_encoder:
        raise ValueError("this model requires a CAM++ ONNX path via speaker_encoder")
    output = Path(output)
    prepared = prepare_prompt_audio(
        reference_audio, output_dir=output.parent / ".prompt_cache"
    )
    codec = CodecDecoder(codec_dir, device=device)
    prompt = codec.encode_file(prepared.path)
    if not 24 <= len(prompt) <= 144:
        raise ValueError(
            f"reference must contain 24–144 codec frames; found {len(prompt)}"
        )
    encoded = [
        tokenizer(t.strip(), add_special_tokens=False)["input_ids"]
        for t in (reference_text, text)
    ]
    if any(len(ids) > 96 for ids in encoded):
        raise ValueError(
            "reference/target exceed the trained 96-token segment limit; split the text explicitly"
        )
    bos = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    ids = [
        bos,
        *prefix_language(encoded[0], reference_language, config),
        *prefix_language(encoded[1], language, config),
    ]
    vector = None
    if config.speaker_conditioning:
        from .speaker import SpeakerEmbeddingExtractor

        with SpeakerEmbeddingExtractor(
            speaker_encoder, show_progress=False
        ) as extractor:
            vector = (
                torch.as_tensor(extractor.extract_from_file(prepared.path))
                .reshape(1, -1)
                .to(device)
            )
    generator = model
    if backend != "native":
        from .acceleration.fast_generate import FastGenerator

        generator = FastGenerator(
            model,
            device=torch.device(device),
            use_tilelang=True,
            batch_branches=True,
            native_sampling=True,
            use_graph=backend == "graph",
        )
    kwargs = dict(
        input_ids=torch.tensor([ids], device=device),
        prompt_codes=prompt,
        uncond_input_ids=torch.tensor([[bos]], device=device),
        seed=seed,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_new_frames=8,
        max_new_frames=max_new_frames,
    )
    if backend == "native":
        kwargs["speaker_embedding"] = vector
    codes, info = generator.generate(**kwargs)
    if not len(codes):
        raise RuntimeError("model generated no audio frames")
    codec.decode_to_file(codes, output)
    np.save(output.with_suffix(".npy"), codes.cpu().numpy(), allow_pickle=False)
    info.update(
        model_id=metadata["release_id"],
        step=metadata["step"],
        backend=backend,
        reference_language=reference_language,
        language=language,
        seed=seed,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seconds=len(codes) / 12.5,
        sample_rate=48000,
    )
    output.with_suffix(".json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    )
    return info
