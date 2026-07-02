"""Sosie STT — faster-whisper, real-time streaming + one-shot.

WS  /ws          stream 16kHz PCM16 frames -> partial transcripts (live)
                 send text frame "eof" to flush; server replies "[[END]]".
POST /transcribe multipart 'audio' file (fallback) -> {"text": "..."}
GET  /health
Runs on http://localhost:5001

faster-whisper uses CTranslate2 — NO Apple MPS backend, so we run CPU/int8
(fast + tiny). On a GPU box: WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16.
"""
import os
import tempfile

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE * 2  # transcribe every ~2s of speech

app = Flask(__name__)
CORS(app)
sock = Sock(app)

print(f"Loading faster-whisper '{MODEL}' on {DEVICE} ({COMPUTE})...")
model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
print("STT ready.")


def _transcribe(audio):
    segments, _ = model.transcribe(
        audio, language="en", beam_size=1, vad_filter=True
    )
    return " ".join(s.text.strip() for s in segments).strip()


@sock.route("/ws")
def stream(ws):
    buffer = np.zeros((0,), dtype=np.float32)
    while True:
        data = ws.receive()
        if data is None:
            break
        # text control frame
        if isinstance(data, str):
            if data == "eof":
                if len(buffer):
                    text = _transcribe(buffer)
                    if text:
                        ws.send(text)
                ws.send("[[END]]")
                buffer = np.zeros((0,), dtype=np.float32)
            continue
        # binary: raw Int16 PCM @ 16kHz mono
        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buffer = np.concatenate([buffer, chunk])
        if len(buffer) >= CHUNK_SAMPLES:
            segment, buffer = buffer[:CHUNK_SAMPLES], buffer[CHUNK_SAMPLES:]
            try:
                text = _transcribe(segment)
                if text:
                    ws.send(text)
            except Exception as e:
                print("transcribe error:", e)


@app.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return jsonify(error="missing 'audio' file"), 400
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        request.files["audio"].save(tmp.name)
        text = _transcribe(tmp.name)
    return jsonify(text=text)


@app.get("/")
def index():
    return (
        "Sosie STT is running. This is an API only (WS /ws, POST /transcribe, "
        "GET /health).<br>Open the voice UI at "
        '<a href="http://localhost:8000">http://localhost:8000</a> (run '
        "<code>python serve.py</code> from the project root)."
    )


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL, device=DEVICE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)))
