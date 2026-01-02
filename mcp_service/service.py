import os
import uuid
import shutil
from typing import Dict, Any

import httpx
import torch
import torchaudio
from mcp.server import FastMCP

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.getenv("COSYVOICE3_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
AUDIO_IN_DIR = os.getenv("AUDIO_FILE_DIR", "audio_file")
AUDIO_OUT_DIR = os.getenv("AUDIO_FILE_GEN_DIR", "audio_file_gen")


mcp = FastMCP(name="CosyVoice3MCP")
cosyvoice = AutoModel(model_dir=MODEL_DIR)


def _ensure_dirs() -> None:
    os.makedirs(AUDIO_IN_DIR, exist_ok=True)
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)


def _normalize_prompt_text(prompt_text: str) -> str:
    if "<|endofprompt|>" not in prompt_text:
        return f"{prompt_text}<|endofprompt|>"
    return prompt_text


def _copy_to_audio_in(path_value: str) -> str:
    _ensure_dirs()
    suffix = os.path.splitext(path_value)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dst_path = os.path.join(AUDIO_IN_DIR, filename)
    shutil.copy2(path_value, dst_path)
    return dst_path


def _download_to_audio_in(url: str) -> str:
    _ensure_dirs()
    suffix = os.path.splitext(url.split("?")[0])[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dst_path = os.path.join(AUDIO_IN_DIR, filename)
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            f.write(resp.content)
    return dst_path


def _resolve_prompt_audio(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return _download_to_audio_in(path_or_url)
    abs_path = os.path.abspath(path_or_url)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"prompt audio not found: {abs_path}")
    return _copy_to_audio_in(abs_path)


def _collect_audio(gen) -> torch.Tensor:
    chunks = []
    for out in gen:
        chunks.append(out["tts_speech"].cpu())
    if not chunks:
        raise RuntimeError("no audio returned from model")
    return torch.cat(chunks, dim=1)


def _save_audio(audio: torch.Tensor) -> str:
    _ensure_dirs()
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.abspath(os.path.join(AUDIO_OUT_DIR, out_name))
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    return out_path


@mcp.tool(
    name="cosyvoice3_zero_shot",
    description="Zero-shot voice clone using local path or URL prompt audio.",
)
async def cosyvoice3_zero_shot(
    text: str,
    prompt_text: str,
    prompt_wav_path: str,
    speed: float = 1.0,
) -> Dict[str, Any]:
    prompt_path = _resolve_prompt_audio(prompt_wav_path)
    prompt_text = _normalize_prompt_text(prompt_text)
    audio = _collect_audio(
        cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            prompt_path,
            stream=False,
            speed=speed,
        )
    )
    out_path = _save_audio(audio)
    return {
        "status": "success",
        "audio_path": out_path,
        "prompt_audio_path": os.path.abspath(prompt_path),
        "sample_rate": cosyvoice.sample_rate,
    }


@mcp.tool(
    name="cosyvoice3_cross_lingual",
    description="Cross-lingual voice clone using local path or URL prompt audio.",
)
async def cosyvoice3_cross_lingual(
    text: str,
    prompt_wav_path: str,
    speed: float = 1.0,
) -> Dict[str, Any]:
    prompt_path = _resolve_prompt_audio(prompt_wav_path)
    audio = _collect_audio(
        cosyvoice.inference_cross_lingual(
            text,
            prompt_path,
            stream=False,
            speed=speed,
        )
    )
    out_path = _save_audio(audio)
    return {
        "status": "success",
        "audio_path": out_path,
        "prompt_audio_path": os.path.abspath(prompt_path),
        "sample_rate": cosyvoice.sample_rate,
    }


@mcp.tool(
    name="cosyvoice3_instruct",
    description="Instruct-based voice clone using local path or URL prompt audio.",
)
async def cosyvoice3_instruct(
    text: str,
    instruct_text: str,
    prompt_wav_path: str,
    speed: float = 1.0,
) -> Dict[str, Any]:
    prompt_path = _resolve_prompt_audio(prompt_wav_path)
    instruct_text = _normalize_prompt_text(instruct_text)
    audio = _collect_audio(
        cosyvoice.inference_instruct2(
            text,
            instruct_text,
            prompt_path,
            stream=False,
            speed=speed,
        )
    )
    out_path = _save_audio(audio)
    return {
        "status": "success",
        "audio_path": out_path,
        "prompt_audio_path": os.path.abspath(prompt_path),
        "sample_rate": cosyvoice.sample_rate,
    }
