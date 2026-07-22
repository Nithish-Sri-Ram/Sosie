"""Sosie vid_gen - MuseTalk V1.5 realtime lip-sync over HTTP (CUDA only).

POST /generate_stream  audio wav (raw body) -> fragmented mp4, streamed while
                       frames are still being generated (play via MediaSource)
POST /generate         audio wav (multipart 'audio' or raw body) -> video/mp4
GET  /idle             pre-rendered silent loop of the avatar (idle state)
GET  /health
Runs on http://localhost:5003

Models load once at startup and the avatar image is pre-processed into
latents, so each request only runs UNet + VAE decode plus the ffmpeg mux -
that is what makes MuseTalk realtime. Run with vid_gen/.venv/bin/python
(its own venv: torch 2.2.0/cu121 + OpenMMLab; see SETUP.md).
"""
import os
import shutil
import sys
import threading
import time
import uuid

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE = os.path.dirname(os.path.abspath(__file__))
MUSETALK_DIR = os.getenv("MUSETALK_DIR", os.path.join(BASE, "MuseTalk"))
AVATAR_ID = os.getenv("AVATAR_ID", "elon")
AVATAR_SRC = os.getenv("AVATAR_SRC", os.path.join(BASE, "avatar_src"))
FPS = int(os.getenv("FPS", 25))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 20))
PORT = int(os.getenv("PORT", 5003))

sys.path.insert(0, MUSETALK_DIR)
os.chdir(MUSETALK_DIR)  # MuseTalk resolves ./models, ./results relative to its root

import copy  # noqa: E402
import subprocess  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, request, jsonify, send_file, Response, stream_with_context  # noqa: E402
from flask_cors import CORS  # noqa: E402
from transformers import WhisperModel  # noqa: E402

import scripts.realtime_inference as ri  # noqa: E402
from musetalk.utils.utils import load_all_model, datagen  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_image_blending  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402

# scripts/realtime_inference.py keeps its config and models in module globals;
# populate them exactly as its __main__ block would, then reuse its Avatar.
ri.args = SimpleNamespace(
    version="v15",
    extra_margin=10,
    parsing_mode="jaw",
    skip_save_images=False,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ri.device = device

print("Loading MuseTalk models...")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)
ri.timesteps = torch.tensor([0], device=device)
ri.pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
ri.vae, ri.unet = vae, unet
ri.weight_dtype = unet.model.dtype
ri.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
ri.whisper = whisper.to(device=device, dtype=unet.model.dtype).eval()
ri.whisper.requires_grad_(False)
ri.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

# A half-finished preparation (crash mid-way) leaves a dir without latents -
# wipe it so Avatar re-prepares instead of failing to load.
avatar_dir = f"./results/v15/avatars/{AVATAR_ID}"
if os.path.isdir(avatar_dir) and not os.path.exists(os.path.join(avatar_dir, "latents.pt")):
    shutil.rmtree(avatar_dir)

avatar = ri.Avatar(
    avatar_id=AVATAR_ID,
    video_path=AVATAR_SRC,
    bbox_shift=0,
    batch_size=BATCH_SIZE,
    preparation=not os.path.isdir(avatar_dir),
)
lock = threading.Lock()

IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", 8))
idle_path = os.path.abspath(f"./results/idle_{AVATAR_ID}.mp4")


def _ensure_idle():
    """Render the avatar over silence once so the UI has an 'alive' idle loop."""
    if os.path.exists(idle_path):
        return
    print("Rendering idle loop...")
    import wave

    silence = os.path.abspath("./results/idle_silence.wav")
    with wave.open(silence, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000 * IDLE_SECONDS)
    avatar.inference(silence, "idle_tmp", FPS, False)
    shutil.move(os.path.abspath(os.path.join(avatar.video_out_path, "idle_tmp.mp4")), idle_path)
    os.remove(silence)
    print("Idle loop ready.")


_ensure_idle()
print("vid_gen ready.")

app = Flask(__name__)
CORS(app)


@app.get("/idle")
def idle():
    if not os.path.exists(idle_path):
        return jsonify(error="idle loop not ready"), 404
    return send_file(idle_path, mimetype="video/mp4")


def _frames(audio_path):
    """Yield blended full frames (BGR uint8) as MuseTalk generates them."""
    weight_dtype = ri.unet.model.dtype
    whisper_input_features, librosa_length = ri.audio_processor.get_audio_feature(
        audio_path, weight_dtype=weight_dtype
    )
    whisper_chunks = ri.audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, ri.whisper, librosa_length,
        fps=FPS, audio_padding_length_left=2, audio_padding_length_right=2,
    )
    gen = datagen(whisper_chunks, avatar.input_latent_list_cycle, BATCH_SIZE)
    idx = 0
    n = len(avatar.coord_list_cycle)
    with torch.no_grad():
        for whisper_batch, latent_batch in gen:
            audio_feat = ri.pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=weight_dtype)
            pred_latents = ri.unet.model(
                latent_batch, ri.timesteps, encoder_hidden_states=audio_feat
            ).sample
            recon = ri.vae.decode_latents(pred_latents.to(device=device, dtype=ri.vae.vae.dtype))
            for res_frame in recon:
                x1, y1, x2, y2 = avatar.coord_list_cycle[idx % n]
                ori = copy.deepcopy(avatar.frame_list_cycle[idx % n])
                try:
                    res = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    idx += 1
                    continue
                mask = avatar.mask_list_cycle[idx % n]
                mask_box = avatar.mask_coords_list_cycle[idx % n]
                frame = get_image_blending(ori, res, [x1, y1, x2, y2], mask, mask_box)
                yield np.ascontiguousarray(frame)
                idx += 1


@app.post("/generate_stream")
def generate_stream():
    payload = request.get_data()
    if not payload:
        return jsonify(error="missing audio (raw wav body)"), 400

    job = uuid.uuid4().hex[:8]
    audio_path = os.path.abspath(f"./results/sosie_{job}.wav")
    with open(audio_path, "wb") as f:
        f.write(payload)

    h, w = avatar.frame_list_cycle[0].shape[:2]
    cmd = [
        "ffmpeg", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-i", audio_path,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
        "-c:a", "aac", "-b:a", "96k",
        # fragmented mp4 so the browser (MediaSource) can play while we generate
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-frag_duration", "500000",
        "-shortest", "-f", "mp4", "pipe:1",
    ]

    def stream():
        with lock:  # one generation at a time on the GPU
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            def feed():
                try:
                    for frame in _frames(audio_path):
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

    job = uuid.uuid4().hex[:8]
    audio_path = os.path.abspath(f"./results/sosie_{job}.wav")
    with open(audio_path, "wb") as f:
        f.write(payload)
    try:
        with lock:  # one generation at a time on the GPU
            # drop clips from previous turns so vid_output does not grow forever
            for old in os.listdir(avatar.video_out_path):
                try:
                    os.remove(os.path.join(avatar.video_out_path, old))
                except OSError:
                    pass
            t0 = time.time()
            avatar.inference(audio_path, job, FPS, skip_save_images=False)
            print(f"job {job}: generated in {time.time() - t0:.1f}s")
        # absolute: Flask's send_file resolves relative paths against app.root_path
        out = os.path.abspath(os.path.join(avatar.video_out_path, f"{job}.mp4"))
        if not os.path.exists(out):
            return jsonify(error="musetalk produced no output"), 500
        return send_file(out, mimetype="video/mp4")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/health")
def health():
    return jsonify(status="ok", avatar=AVATAR_ID, device=str(device))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
