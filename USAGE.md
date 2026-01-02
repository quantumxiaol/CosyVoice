# Usage

This document summarizes how to start services and run local test scripts.

## Model and audio files

- Model dir (default): `pretrained_models/Fun-CosyVoice3-0.5B`
- Prompt audio for tests: `audio_file/Mihono_Bourbon.mp3`
- Generated audio output: `audio_file_gen/`

## Services

### FastAPI service (CosyVoice3)

Start:

```bash
python fastapi_service/server.py --host 0.0.0.0 --port 8891
```

Local path usage (form fields):

- `prompt_wav_path`: absolute or relative path to local file
- Output path in response is absolute and stored under `audio_file_gen/`

### MCP server (CosyVoice3)

Start (HTTP/SSE):

```bash
python mcp_service/server.py --http --host 0.0.0.0 --port 8890
```

MCP tools (local path or URL prompt audio):

- `cosyvoice3_zero_shot`
- `cosyvoice3_cross_lingual`
- `cosyvoice3_instruct`

All tools return `audio_path` as an absolute path under `audio_file_gen/`.

## Tests

### CosyVoice3 direct inference

```bash
uv run python tests/test_cosyvoice3.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --prompt-wav audio_file/Mihono_Bourbon.mp3 \
  --mode cross_lingual \
  --text-zh "这是中文测试句子，用来检查音色和韵律。" \
  --text-ja "これは日本語のテスト文です。音色と韻律を確認します。"
```

Outputs go to `audio_file_gen/` with unique filenames.

### FastAPI test

Ensure the FastAPI service is running, then:

```bash
uv run python tests/test_fastapi.py \
  --base-url http://127.0.0.1:8891 \
  --mode cross_lingual \
  --prompt-wav audio_file/Mihono_Bourbon.mp3 \
  --text "这是中文测试句子，用来检查音色和韵律。"
```

### MCP tool test

Ensure the MCP server is running, then:

```bash
uv run python tests/test_mcp_tool_call.py \
  --base_url http://127.0.0.1:8890/mcp \
  --tool-name cosyvoice3_cross_lingual \
  --prompt-wav-path audio_file/Mihono_Bourbon.mp3 \
  --text "这是中文测试句子，用来检查音色和韵律。"
```
