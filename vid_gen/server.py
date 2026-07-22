"""Sosie vid_gen - Ditto full-face audio-driven avatar over HTTP (CUDA only).

POST /generate_stream?persona=<id>  audio wav (raw body) -> fragmented mp4,
                                    streamed while frames are still generating
                                    (play via MediaSource)
POST /generate?persona=<id>        audio wav (multipart 'audio' or raw body)
                                    -> video/mp4
GET  /idle?persona=<id>            pre-rendered silent loop of the avatar
GET  /personas                     [{"id":.., "name":..}, ...] - available personas
GET  /health
Runs on http://localhost:5003

Ditto (antgroup/ditto-talkinghead) replaces MuseTalk here: it generates full
head motion + expression from a single reference image + audio, not just
lip-sync onto a fixed driving video. The SDK's own frame writer is swapped
for a queue-based one (QueueWriter below) so we can pipe raw frames into our
own ffmpeg process for fragmented-mp4 streaming, same approach the old
MuseTalk server used. Avatar registration (source2info) happens per-request
from personas/<id>/avatar.jpg - cheap (~0.05-0.3s), no pre-baking needed, so
switching characters between turns is free. See ditto-talkinghead-benchmark
memory for perf numbers (A100: ~1.3x real-time for this online/streaming
config). Run with vid_gen/ditto/repo/.venv/bin/python.
"""
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import wave

import imageio_ffmpeg
import librosa
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

DITTO_DIR = os.getenv("DITTO_DIR", os.path.join(BASE, "ditto", "repo"))
sys.path.insert(0, DITTO_DIR)
os.chdir(DITTO_DIR)  # ditto resolves ./checkpoints relative to its root

from stream_pipeline_online import StreamSDK  # noqa: E402

PERSONAS_DIR = os.path.join(BASE, "..", "personas")
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA", "elon")
FPS = int(os.getenv("FPS", 25))
PORT = int(os.getenv("PORT", 5003))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", 8))
CHUNKSIZE = (3, 5, 2)  # (lookback, step, lookahead) frames, per Ditto's online API

TMP_DIR = os.path.join(DITTO_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

DATA_ROOT = os.path.join(DITTO_DIR, "checkpoints", "ditto_trt_Ampere_Plus")
CFG_PKL = os.path.join(DITTO_DIR, "checkpoints", "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")

# resolved explicitly - the system has no ffmpeg on PATH, only this bundled binary
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

print("Loading Ditto...")
sdk = StreamSDK(CFG_PKL, DATA_ROOT)
lock = threading.Lock()  # one generation at a time on the GPU
print("vid_gen (Ditto) ready.")

app = Flask(__name__)
CORS(app)


def avatar_path(persona_id):
    p = os.path.join(PERSONAS_DIR, persona_id or DEFAULT_PERSONA, "avatar.jpg")
    return p if os.path.exists(p) else os.path.join(PERSONAS_DIR, DEFAULT_PERSONA, "avatar.jpg")


class QueueWriter:
    """Drop-in for core.atomic_components.writer.VideoWriterByImageIO that
    pushes raw BGR frames to a queue instead of encoding to disk itself."""

    def __init__(self, out_queue):
        self.out_queue = out_queue

    def __call__(self, img, fmt="bgr"):
        frame = img if fmt == "bgr" else img[..., ::-1]
        self.out_queue.put(np.ascontiguousarray(frame))

    def close(self):
        self.out_queue.put(None)  # sentinel: no more frames


def _frames(source_path, audio_path):
    """Yield raw BGR full frames (uint8) as Ditto generates them."""
    audio, _ = librosa.core.load(audio_path, sr=16000)
    num_f = max(1, round(len(audio) / 16000 * FPS))
    out_q = queue.Queue()

    tmp_out = os.path.join(TMP_DIR, f"unused_{uuid.uuid4().hex[:8]}.mp4")
    sdk.setup(source_path, tmp_out)
    sdk.writer = QueueWriter(out_q)  # override after setup() builds its own
    sdk.setup_Nd(N_d=num_f)

    def feed():
        try:
            padded = np.concatenate(
                [np.zeros((CHUNKSIZE[0] * 640,), dtype=np.float32), audio], 0
            )
            split_len = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
            for i in range(0, len(padded), CHUNKSIZE[1] * 640):
                chunk = padded[i : i + split_len]
                if len(chunk) < split_len:
                    chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
                sdk.run_chunk(chunk, CHUNKSIZE)
        finally:
            sdk.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        while True:
            frame = out_q.get()
            if frame is None:
                break
            yield frame
    finally:
        feeder.join(timeout=10)


def _ffmpeg_mux_cmd(w, h, audio_path, out, fragmented=True):
    cmd = [
        FFMPEG_BIN, "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-i", audio_path,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
        "-c:a", "aac", "-b:a", "96k",
    ]
    if fragmented:
        # fragmented mp4 so the browser (MediaSource) can play while we generate
        cmd += ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "500000"]
    return cmd + ["-shortest", "-f", "mp4", out]


@app.get("/personas")
def personas():
    out = []
    for pdir in sorted(glob.glob(os.path.join(PERSONAS_DIR, "*"))):
        pj = os.path.join(pdir, "persona.json")
        if not os.path.exists(pj):
            continue
        try:
            with open(pj) as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            continue
        out.append({"id": os.path.basename(pdir), "name": meta.get("name", os.path.basename(pdir))})
    return jsonify(out)


def _idle_path(persona_id):
    return os.path.join(TMP_DIR, f"idle_{persona_id}.mp4")


def _ensure_idle(persona_id):
    path = _idle_path(persona_id)
    if os.path.exists(path):
        return path
    print(f"Rendering idle loop for {persona_id}...")
    silence_path = os.path.join(TMP_DIR, f"idle_silence_{persona_id}.wav")
    with wave.open(silence_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * IDLE_SECONDS)
    with lock:
        gen = _frames(avatar_path(persona_id), silence_path)
        first = next(gen, None)
        if first is None:
            raise RuntimeError("ditto produced no idle frames")
        h, w = first.shape[:2]
        proc = subprocess.Popen(
            _ffmpeg_mux_cmd(w, h, silence_path, path, fragmented=False),
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        try:
            proc.stdin.write(first.tobytes())
            for frame in gen:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        proc.wait()
    os.remove(silence_path)
    print(f"Idle loop ready for {persona_id}.")
    return path


@app.get("/idle")
def idle():
    persona = request.args.get("persona", DEFAULT_PERSONA)
    try:
        path = _ensure_idle(persona)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(path, mimetype="video/mp4")


@app.post("/generate_stream")
def generate_stream():
    payload = request.get_data()
    if not payload:
        return jsonify(error="missing audio (raw wav body)"), 400
    persona = request.args.get("persona", DEFAULT_PERSONA)
    src = avatar_path(persona)

    job = uuid.uuid4().hex[:8]
    audio_path = os.path.abspath(os.path.join(TMP_DIR, f"sosie_{job}.wav"))
    with open(audio_path, "wb") as f:
        f.write(payload)

    def stream():
        with lock:
            gen = _frames(src, audio_path)
            try:
                first = next(gen)
            except StopIteration:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return
            h, w = first.shape[:2]
            proc = subprocess.Popen(
                _ffmpeg_mux_cmd(w, h, audio_path, "pipe:1"),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

            def feed():
                try:
                    proc.stdin.write(first.tobytes())
                    for frame in gen:
                        proc.stdin.write(frame.tobytes())
                except Exception as e:
                    print(f"job {job}: frame feed error: {e}")
                finally:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass

            t0 = time.time()
            feeder = threading.Thread(target=feed, daemon=True)
            feeder.start()
            try:
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break
                    yield chunk
                print(f"job {job}: streamed in {time.time() - t0:.1f}s")
            finally:
                feeder.join(timeout=10)
                proc.kill()
                if os.path.exists(audio_path):
                    os.remove(audio_path)

    return Response(
        stream_with_context(stream()),
        mimetype="video/mp4",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/generate")
def generate():
    if "audio" in request.files:
        payload = request.files["audio"].read()
    else:
        payload = request.get_data()
    if not payload:
        return jsonify(error="missing audio (multipart 'audio' or raw wav body)"), 400
    persona = request.args.get("persona", DEFAULT_PERSONA)
    src = avatar_path(persona)

    job = uuid.uuid4().hex[:8]
    audio_path = os.path.abspath(os.path.join(TMP_DIR, f"sosie_{job}.wav"))
    out_path = os.path.abspath(os.path.join(TMP_DIR, f"sosie_{job}.mp4"))
    with open(audio_path, "wb") as f:
        f.write(payload)
    try:
        with lock:
            t0 = time.time()
            gen = _frames(src, audio_path)
            first = next(gen, None)
            if first is None:
                return jsonify(error="ditto produced no output"), 500
            h, w = first.shape[:2]
            cmd = _ffmpeg_mux_cmd(w, h, audio_path, out_path, fragmented=False)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                proc.stdin.write(first.tobytes())
                for frame in gen:
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                # ffmpeg's -shortest exits once the (shorter) audio track ends;
                # Ditto's online chunking can emit a handful of frames past
                # that point - harmless, just stop feeding them.
                pass
            proc.wait()
            print(f"job {job}: generated in {time.time() - t0:.1f}s")
        if not os.path.exists(out_path):
            return jsonify(error="ditto produced no output"), 500
        return send_file(out_path, mimetype="video/mp4")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(out_path):
            threading.Timer(30, lambda: os.path.exists(out_path) and os.remove(out_path)).start()


@app.get("/health")
def health():
    return jsonify(status="ok", engine="ditto", device="cuda")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
