"""Sosie TTS - CosyVoice2-0.5B over HTTP.

POST /tts  {"text": "..."}  -> audio/wav
GET  /health
Runs on http://localhost:5002

Setup (see README.md): clone CosyVoice into ./CosyVoice, download the
CosyVoice2-0.5B weights, and drop a short reference clip at ./assets/prompt.wav.
Prefers Apple MPS, falls back to CPU; uses CUDA automatically on a GPU box.
"""
import io
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

app = Flask(__name__)
CORS(app)

print("Loading CosyVoice2-0.5B...")
cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
if not os.path.exists(PROMPT_WAV):
    raise SystemExit(f"Reference voice clip not found: {PROMPT_WAV}")
print("TTS ready.")


@app.post("/tts")
def tts():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="missing 'text'"), 400
    chunks = [
        out["tts_speech"]
        for out in cosyvoice.inference_zero_shot(
            text, PROMPT_TEXT, PROMPT_WAV, stream=False
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
