# Soraku TTS

基于 Qwen3 骨干的 48kHz TTS 独立推理实现，提供中文及中英日多语模型的本地加载与参考音频合成。

独立推理代码。模型权重、文本tokenizer、音频codec、CAM++均从单独的模型目录加载。
不需要原研究目录、训练数据、ASR服务或额外Qwen3基座权重；不需要 `trust_remote_code`。
这里使用专用加载器，不是 Transformers `AutoModel.from_pretrained` 的注册模型。

## 安装

已核验环境为Linux、Python3.13、PyTorch2.14.1、Transformers4.57.3、qwen-tts0.1.1。
先安装适配本机CUDA的PyTorch/torchaudio，再在本目录执行：

```bash
python -m pip install -e '.[speaker]'
```

`speaker`用于D模型；A/中文102k的native路径不要求ONNX。
`fast`为可选TileLang依赖；D仅支持native。不同GPU/浮点快路径同seed不保证整条tokens一致。

## 推理

模型由使用者通过 ModelScope SDK 下载，本仓库只提供代码，不包含权重、文本 tokenizer、codec 或 CAM++ 文件，也不自动下载模型。
将下载后的完整模型目录放在任意位置，通过下面的路径参数指定即可。以下按代码与 `models/` 同级的布局举例，在本目录运行：

```bash
soraku-infer \
  --model-dir ../models/soraku-multilingual-d-60k \
  --codec-dir ../models/qwen3-tts-tokenizer-12hz-48khz \
  --speaker-encoder ../models/campplus/campplus.onnx \
  --reference-audio /path/to/reference.wav \
  --reference-text '参考音频的准确逐字稿。' \
  --reference-language zh --language zh \
  --text '今天我们来测试语音合成。' \
  --seed 42 --cfg-scale 1.5 --output output.wav
```

输出WAV（48kHz）、同名NPY（16码本tokens）、JSON参数与终止状态。
参考自动按训练口径单声道、24kHz、去头/限制尾静音、RMS−23dB预处理。
参考范围24–144帧（约1.92–11.52秒），推荐提供清晰、内容准确的4–6秒语音；文本段不超过96tokens。
四个语言标记 `<|lang_zh|>` / `<|lang_en|>` / `<|lang_ja|>` / `<|lang_auto|>` 已随多语模型tokenizer保存。
不要手工改ID或把不同模型tokenizer互换。参考语言优先明确填写实际语言；日文目标优先ja。

## 版本与已知限制

- `soraku-zh-102k`：中文模型，使用 `--language zh`；不支持用它评价英文/日文。
- `soraku-multilingual-a-60k`：中英日研究版本，原v2路线A。
- `soraku-multilingual-d-60k`：中英日研究版本，路线D含prompt-aligned CAM++条件，需要额外的CAM++ ONNX文件。
- A/D分别提供不同条件设计，不自动指定全局最佳版本。50k不在本轮发布范围。
- 跨语音色、语速/口音、日文汉字首词和三语混合仍可能失稳。auto不保证正确选择发音。
- 合法tokens或正常EOS不代表无电音/遗漏。分段、显式语言与假名读音属于候选缓解，不能保证修复。
- 不自动请求ASR；参考逐字稿由调用者提供。默认native FP32、CFG1.5、temperature0.9、top-k50/top-p0.95。

## 模型与依赖

每个Soraku模型应包含 `model.safetensors`、`config.json`、`text_config.json` 和完整 `tokenizer/`。权重包含完整Qwen3主干，无须再下载一份Qwen3基座权重。模型的ModelScope仓库地址由发布者另行提供，不使用猜测的仓库ID。

- 音频codec：[takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz](https://huggingface.co/takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz)，所有版本均需要。
- D的speaker encoder：[CosyVoice2-0.5B/campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/campplus.onnx)，SHA256为 `a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73`。

公开代码仓库：[SoraKuAI/soraku-TTS](https://github.com/SoraKuAI/soraku-TTS)。代码不包含训练数据、参考录音、研究工程快照或上传凭据。

## 许可

本发布仓库沿用初始化时的Apache-2.0许可，见 `LICENSE`。来自原研究工程的代码保留MIT许可声明，见 `ORIGINAL_MIT_LICENSE`。第三方依赖和模型权重的许可单独列于 `THIRD_PARTY_NOTICES.md` 及各模型卡；代码许可不替模型权重作授权声明。
