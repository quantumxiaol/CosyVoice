# CosyVoice3 用法与效果速览

本文件整理 CosyVoice3 的常见调用方式、效果特点与注意事项，便于快速上手与对外说明。

## 1. 核心能力（效果）

- 零样本语音克隆：仅凭一段参考音频即可复刻说话人音色。
- 跨语种复刻：参考音频是 A 语言，合成文本可为 B 语言。
- 自然语言指令控制：通过指令控制语速、情绪、方言等。
- 细粒度控制：支持在文本中插入控制标记（如 ` [breath] ` 等）。
- 文本规范化：对数字、符号、格式化文本具备一定自处理能力。

## 2. 关键输入约束

- CosyVoice3 要求 prompt 或 instruct 文本包含 `<|endofprompt|>`。
- 零样本/跨语种的 prompt 音频必须与 prompt 文本内容一致。
- 建议 prompt 音频采样率不低于 16 kHz。

## 3. Python 直接调用（示例）

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# 零样本克隆
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        '八百标兵奔北坡，北坡炮兵并排跑。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False,
    )
):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 跨语种复刻
for i, j in enumerate(
    cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>Hello, how are you today?',
        './asset/zero_shot_prompt.wav',
        stream=False,
    )
):
    torchaudio.save(f'cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 指令控制
for i, j in enumerate(
    cosyvoice.inference_instruct2(
        '收到音频后请用广东话表达。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        './asset/zero_shot_prompt.wav',
        stream=False,
    )
):
    torchaudio.save(f'instruct_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 4. FastAPI 服务调用方式（本仓库）

服务启动：

```bash
python fastapi_service/server.py --host 0.0.0.0 --port 8891
```

零样本克隆：

```bash
curl -X POST "http://127.0.0.1:8891/tts/zero_shot" \
  -F "text=八百标兵奔北坡，北坡炮兵并排跑。" \
  -F "prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  -F "speed=1.0"
```

跨语种复刻：

```bash
curl -X POST "http://127.0.0.1:8891/tts/cross_lingual" \
  -F "text=You are a helpful assistant.<|endofprompt|>Hello, how are you today?" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  -F "speed=1.0"
```

指令控制：

```bash
curl -X POST "http://127.0.0.1:8891/tts/instruct" \
  -F "text=收到音频后请用广东话表达。" \
  -F "instruct_text=You are a helpful assistant. 请用广东话表达。<|endofprompt|>" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  -F "speed=1.0"
```

输出文件位置：

- 参考音频保存目录：`audio_file/`
- 合成音频输出目录：`audio_file_gen/`
- 下载接口：`GET /audio/{filename}`
