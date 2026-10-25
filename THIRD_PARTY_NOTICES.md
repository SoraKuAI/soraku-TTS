# Third-party components and retained notices

This inference distribution retains the Apache License 2.0 supplied by the initialized SoraKuAI/soraku-TTS publication repository (see LICENSE).

Inference code derived from the original Soraku research repository retains its MIT notice, Copyright (c) 2026 NorthSky, in ORIGINAL_MIT_LICENSE. This notice must remain with copies or substantial portions of that code.

Model weights and tokenizers are downloaded separately and are not included in this code repository:

- Qwen3 tensors and text tokenizer originate from [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base); retain the upstream Apache-2.0 license supplied with each model folder.
- The external 48kHz audio codec is [takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz](https://huggingface.co/takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz), not a Soraku-trained codec. Refer to its original model card and license.
- The CAM++ speaker encoder used by D is [FunAudioLLM/CosyVoice2-0.5B/campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/campplus.onnx). The official model card declares Apache-2.0. The file is 28,303,423 bytes with SHA256 `a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73`, matching the upstream LFS pointer. The ONNX binary is not included here.

No code license in this repository assigns a license to Soraku-trained model weights. Consult each model's separately supplied license and model card.
