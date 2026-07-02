"""Sosie STT - faster-whisper + Silero VAD turn detection.

WS  /ws          stream 16kHz PCM16 frames continuously. The server detects
                 turns with Silero VAD: "[[S]]" marks speech onset, live
                 partials arrive as "[[P]]<text>" while you speak, and once
                 you pause the final transcript is sent followed by "[[END]]".
                 A text frame "eof" force-flushes the current utterance.
POST /transcribe multipart 'audio' file (fallback) -> {"text": "..."}
GET  /health
Runs on http://localhost:5001

faster-whisper uses CTranslate2 - NO Apple MPS backend, so we run CPU/int8
(fast + tiny). On a GPU box: WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16.
"""
import os
import tempfile

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

SAMPLE_RATE = 16000
VAD_WINDOW = 512                    # samples per VAD step (Silero @16k)
PREROLL = SAMPLE_RATE // 2          # keep 0.5s of audio before speech onset
PARTIAL_EVERY = SAMPLE_RATE * 2     # live partial transcript every ~2s
END_SILENCE_MS = 700                # pause length that ends a turn
MIN_UTTERANCE = SAMPLE_RATE // 4    # ignore blips shorter than 0.25s

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
    # per-connection VAD: Silero keeps LSTM state inside the model object
    vad = VADIterator(load_silero_vad(), sampling_rate=SAMPLE_RATE,
                      min_silence_duration_ms=END_SILENCE_MS, speech_pad_ms=100)
    stash = np.zeros(0, np.float32)      # samples not yet fed to VAD
    preroll = np.zeros(0, np.float32)    # rolling buffer of pre-speech audio
    utterance = []                       # windows collected since speech onset
    talking = False
    since_partial = 0

    def flush():
        nonlocal utterance, talking, since_partial
        audio = np.concatenate(utterance) if utterance else np.zeros(0, np.float32)
        utterance, talking, since_partial = [], False, 0
        vad.reset_states()
        if len(audio) >= MIN_UTTERANCE:
            try:
                text = _transcribe(audio)
                if text:
                    ws.send(text)
            except Exception as e:
                print("transcribe error:", e)
        ws.send("[[END]]")

    while True:
        data = ws.receive()
        if data is None:
            break
        # text control frame
        if isinstance(data, str):
            if data == "eof" and talking:
                flush()
            continue
        # binary: raw Int16 PCM @ 16kHz mono
        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        stash = np.concatenate([stash, chunk])
        while len(stash) >= VAD_WINDOW:
            win, stash = stash[:VAD_WINDOW], stash[VAD_WINDOW:]
            try:
                event = vad(torch.from_numpy(win))
            except Exception as e:
                print("vad error:", e)
                event = None
            if talking:
                utterance.append(win)
                since_partial += VAD_WINDOW
            else:
                preroll = np.concatenate([preroll, win])[-PREROLL:]
            if event and "start" in event and not talking:
                talking = True
                utterance = [preroll, win]
                preroll = np.zeros(0, np.float32)
                since_partial = 0
                ws.send("[[S]]")
            elif event and "end" in event and talking:
                flush()
            elif talking and since_partial >= PARTIAL_EVERY:
                since_partial = 0
                try:
                    text = _transcribe(np.concatenate(utterance))
                    if text:
                        ws.send("[[P]]" + text)
                except Exception as e:
                    print("partial transcribe error:", e)


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
