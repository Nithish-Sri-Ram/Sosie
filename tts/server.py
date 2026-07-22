"""Sosie TTS - CosyVoice2-0.5B over HTTP.

POST /tts  {"text": "...", "persona": "elon"}  -> audio/wav
GET  /health
Runs on http://localhost:5002

Setup (see README.md): clone CosyVoice into ./CosyVoice, download the
CosyVoice2-0.5B weights, and drop a short reference clip at ./assets/prompt.wav.
Prefers Apple MPS, falls back to CPU; uses CUDA automatically on a GPU box.

Voice reference (prompt wav + its transcript) is resolved per-request from
personas/<id>/{voice.wav,persona.json's voice_text}, falling back to the
PROMPT_WAV/PROMPT_TEXT .env pair when a persona isn't found - CosyVoice2's
zero-shot cloning takes the prompt path/text per call, so no model reload
is needed to switch voices.
"""
import io
import json
import os
import sys

import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Make the cloned CosyVoice repo importable
COSYVOICE_DIR = os.getenv(
    "COSYVOICE_DIR", os.path.join(os.path.dirname(__file__), "CosyVoice")
)
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2       # noqa: E402
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Can't import CosyVoice ({e}). It is NOT a pip package - clone the repo "
        f"into {COSYVOICE_DIR} and install its deps. See tts/SETUP.md:\n"
        "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
        "  pip install -r CosyVoice/requirements.txt"
    )

MODEL_DIR = os.getenv(
    "COSYVOICE_MODEL",
    os.path.join(COSYVOICE_DIR, "pretrained_models", "CosyVoice2-0.5B"),
)
PROMPT_WAV = os.getenv(
    "PROMPT_WAV", os.path.join(os.path.dirname(__file__), "assets", "prompt.wav")
)
PROMPT_TEXT = os.getenv("PROMPT_TEXT", "Hello, this is a reference voice sample.")
PERSONAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "personas")


def voice_ref(persona_id):
    """(prompt_wav_path, prompt_text) for personas/<id>, falling back to .env's default."""
    pdir = os.path.join(PERSONAS_DIR, persona_id or "")
    wav = os.path.join(pdir, "voice.wav")
    try:
        with open(os.path.join(pdir, "persona.json")) as f:
            text = json.load(f)["voice_text"]
    except (OSError, KeyError, json.JSONDecodeError):
        return PROMPT_WAV, PROMPT_TEXT
    return (wav if os.path.exists(wav) else PROMPT_WAV), text


app = Flask(__name__)
CORS(app)

print("Loading CosyVoice2-0.5B...")
cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
if not os.path.exists(PROMPT_WAV):
    raise SystemExit(f"Reference voice clip not found: {PROMPT_WAV}")
print("TTS ready.")


@app.post("/tts")
def tts():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify(error="missing 'text'"), 400
    prompt_wav, prompt_text = voice_ref(body.get("persona"))
    chunks = [
        out["tts_speech"]
        for out in cosyvoice.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=False
        )
    ]
    audio = torch.cat(chunks, dim=1).cpu()
    buf = io.BytesIO()
    torchaudio.save(buf, audio, cosyvoice.sample_rate, format="wav")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5002)))
