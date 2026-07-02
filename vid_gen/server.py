"""Sosie vid_gen — MuseTalk lip-sync over HTTP (GPU / CUDA box only).

POST /generate  (multipart: 'audio' wav + optional 'avatar' id) -> video/mp4
GET  /health
Runs on http://localhost:5003

This is a thin wrapper that shells out to MuseTalk's realtime inference script.
It is NOT runnable on this Mac — MuseTalk needs CUDA (see README.md for the
GPU sizing + weight-download steps). Test tts/ and stt/ locally; run this on
the GPU box.
"""
import os
import subprocess
import sys
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

MUSETALK_DIR = os.getenv(
    "MUSETALK_DIR", os.path.join(os.path.dirname(__file__), "MuseTalk")
)
# A prepared avatar/config lives under MuseTalk/configs — set per your setup.
CONFIG = os.getenv("MUSETALK_CONFIG", "configs/inference/realtime.yaml")
RESULT_DIR = os.getenv("RESULT_DIR", os.path.join(MUSETALK_DIR, "results"))

app = Flask(__name__)
CORS(app)


@app.post("/generate")
def generate():
    if "audio" not in request.files:
        return jsonify(error="missing 'audio' file"), 400
    job = uuid.uuid4().hex[:8]
    audio_path = os.path.join(tempfile.gettempdir(), f"sosie_{job}.wav")
    request.files["audio"].save(audio_path)

    # MuseTalk realtime inference: audio_path is injected via the config/CLI.
    cmd = [
        sys.executable, "-m", "scripts.realtime_inference",
        "--inference_config", CONFIG,
        "--audio_path", audio_path,
        "--result_dir", RESULT_DIR,
    ]
    proc = subprocess.run(cmd, cwd=MUSETALK_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        return jsonify(error="musetalk failed", stderr=proc.stderr[-2000:]), 500

    out = os.path.join(RESULT_DIR, f"{job}.mp4")  # adjust to your config's naming
    if not os.path.exists(out):
        return jsonify(error="output not found", stdout=proc.stdout[-2000:]), 500
    return send_file(out, mimetype="video/mp4")


@app.get("/health")
def health():
    return jsonify(status="ok", musetalk=os.path.isdir(MUSETALK_DIR))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5003)))
