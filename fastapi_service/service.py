import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.getenv("COSYVOICE3_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
AUDIO_IN_DIR = os.getenv("AUDIO_FILE_DIR", "audio_file")
AUDIO_OUT_DIR = os.getenv("AUDIO_FILE_GEN_DIR", "audio_file_gen")

cosyvoice = None


def _ensure_dirs() -> None:
    os.makedirs(AUDIO_IN_DIR, exist_ok=True)
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(AUDIO_IN_DIR, filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


def _store_local_prompt(path_value: str) -> str:
    _ensure_dirs()
    abs_path = os.path.abspath(path_value)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"prompt audio not found: {abs_path}")
    suffix = os.path.splitext(abs_path)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dst_path = os.path.join(AUDIO_IN_DIR, filename)
    with open(abs_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(src.read())
    return dst_path


def _collect_audio(gen) -> torch.Tensor:
    chunks = []
    for out in gen:
        chunks.append(out["tts_speech"].cpu())
    if not chunks:
        raise RuntimeError("no audio returned from model")
    return torch.cat(chunks, dim=1)


def _normalize_prompt_text(prompt_text: str) -> str:
    if "<|endofprompt|>" not in prompt_text:
        return f"{prompt_text}<|endofprompt|>"
    return prompt_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cosyvoice
    _ensure_dirs()
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    try:
        yield
    finally:
        cosyvoice = None


app = FastAPI(title="CosyVoice3 Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = os.path.abspath(os.path.join(AUDIO_OUT_DIR, filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path)


@app.post("/tts/zero_shot")
async def tts_zero_shot(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: Optional[UploadFile] = File(None),
    prompt_wav_path: Optional[str] = Form(None),
    speed: float = Form(1.0),
) -> dict:
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if prompt_wav is not None:
        prompt_path = _save_upload(prompt_wav)
    elif prompt_wav_path:
        prompt_path = _store_local_prompt(prompt_wav_path)
    else:
        raise HTTPException(status_code=400, detail="prompt_wav or prompt_wav_path required")
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
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.abspath(os.path.join(AUDIO_OUT_DIR, out_name))
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    return {
        "status": "success",
        "audio_filename": out_name,
        "audio_path": out_path,
        "sample_rate": cosyvoice.sample_rate,
    }


@app.post("/tts/cross_lingual")
async def tts_cross_lingual(
    text: str = Form(...),
    prompt_wav: Optional[UploadFile] = File(None),
    prompt_wav_path: Optional[str] = Form(None),
    speed: float = Form(1.0),
) -> dict:
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if prompt_wav is not None:
        prompt_path = _save_upload(prompt_wav)
    elif prompt_wav_path:
        prompt_path = _store_local_prompt(prompt_wav_path)
    else:
        raise HTTPException(status_code=400, detail="prompt_wav or prompt_wav_path required")

    audio = _collect_audio(
        cosyvoice.inference_cross_lingual(
            text,
            prompt_path,
            stream=False,
            speed=speed,
        )
    )
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.abspath(os.path.join(AUDIO_OUT_DIR, out_name))
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    return {
        "status": "success",
        "audio_filename": out_name,
        "audio_path": out_path,
        "sample_rate": cosyvoice.sample_rate,
    }


@app.post("/tts/instruct")
async def tts_instruct(
    text: str = Form(...),
    instruct_text: str = Form(...),
    prompt_wav: Optional[UploadFile] = File(None),
    prompt_wav_path: Optional[str] = Form(None),
    speed: float = Form(1.0),
) -> dict:
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if prompt_wav is not None:
        prompt_path = _save_upload(prompt_wav)
    elif prompt_wav_path:
        prompt_path = _store_local_prompt(prompt_wav_path)
    else:
        raise HTTPException(status_code=400, detail="prompt_wav or prompt_wav_path required")
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
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.abspath(os.path.join(AUDIO_OUT_DIR, out_name))
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    return {
        "status": "success",
        "audio_filename": out_name,
        "audio_path": out_path,
        "sample_rate": cosyvoice.sample_rate,
    }
