import argparse
import json
from pathlib import Path

import httpx


def _normalize_prompt_text(text: str) -> str:
    if "<|endofprompt|>" not in text:
        return f"{text}<|endofprompt|>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Call CosyVoice3 FastAPI service.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8891",
        help="FastAPI base URL.",
    )
    parser.add_argument(
        "--mode",
        choices=["cross_lingual", "zero_shot", "instruct"],
        default="cross_lingual",
        help="Endpoint mode to test.",
    )
    parser.add_argument(
        "--prompt-wav",
        default="audio_file/Mihono_Bourbon.mp3",
        help="Prompt audio path on local disk.",
    )
    parser.add_argument(
        "--text",
        default="这是中文测试句子，用来检查音色和韵律。",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.",
        help="Prompt text for zero_shot mode.",
    )
    parser.add_argument(
        "--instruct-text",
        default="You are a helpful assistant. 请用广东话表达。",
        help="Instruct text for instruct mode.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Synthesis speed.",
    )
    args = parser.parse_args()

    prompt_wav_path = str(Path(args.prompt_wav).resolve())
    if args.mode == "cross_lingual":
        endpoint = "/tts/cross_lingual"
        payload = {
            "text": args.text,
            "prompt_wav_path": prompt_wav_path,
            "speed": str(args.speed),
        }
    elif args.mode == "zero_shot":
        endpoint = "/tts/zero_shot"
        payload = {
            "text": args.text,
            "prompt_text": _normalize_prompt_text(args.prompt_text),
            "prompt_wav_path": prompt_wav_path,
            "speed": str(args.speed),
        }
    else:
        endpoint = "/tts/instruct"
        payload = {
            "text": args.text,
            "instruct_text": _normalize_prompt_text(args.instruct_text),
            "prompt_wav_path": prompt_wav_path,
            "speed": str(args.speed),
        }

    url = f"{args.base_url}{endpoint}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, data=payload)
        resp.raise_for_status()
        data = resp.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))
    audio_path = data.get("audio_path")
    if audio_path and Path(audio_path).exists():
        print("TEST_RESULT: PASSED")
    else:
        print("TEST_RESULT: FAILED")


if __name__ == "__main__":
    main()
