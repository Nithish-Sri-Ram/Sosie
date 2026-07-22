"""Ditto smoke test - audio -> full-face avatar video, standalone (GPU box).

    python smoke_test.py path/to/audio.wav [path/to/avatar.jpg]

Runs Ditto's StreamSDK directly (same path server.py uses) and times it, so
you get raw audio->video FPS before wiring vid_gen into the pipeline. CUDA
only; this will NOT run on the Mac. Run with
vid_gen/ditto/repo/.venv/bin/python so Ditto's own venv is picked up.
"""
import os
import sys
import time

import librosa
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DITTO_DIR = os.getenv("DITTO_DIR", os.path.join(BASE, "ditto", "repo"))
sys.path.insert(0, DITTO_DIR)
os.chdir(DITTO_DIR)  # ditto resolves ./checkpoints relative to its root

from stream_pipeline_online import StreamSDK  # noqa: E402

audio_path = sys.argv[1] if len(sys.argv) > 1 else "example/audio.wav"
avatar_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    BASE, "..", "personas", "elon", "avatar.jpg"
)
CHUNKSIZE = (3, 5, 2)  # (lookback, step, lookahead) frames, per Ditto's online API

DATA_ROOT = os.path.join(DITTO_DIR, "checkpoints", "ditto_trt_Ampere_Plus")
CFG_PKL = os.path.join(DITTO_DIR, "checkpoints", "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")


class CountingWriter:
    """Drops frames, just counts them - we only want the timing here."""

    def __init__(self):
        self.count = 0

    def __call__(self, img, fmt="bgr"):
        self.count += 1

    def close(self):
        pass


t0 = time.time()
sdk = StreamSDK(CFG_PKL, DATA_ROOT)
print(f"model loaded in {time.time() - t0:.1f}s")

audio, _ = librosa.core.load(audio_path, sr=16000)
num_f = max(1, round(len(audio) / 16000 * 25))

writer = CountingWriter()
tmp_out = os.path.join(DITTO_DIR, "tmp", "smoke_test_unused.mp4")
os.makedirs(os.path.dirname(tmp_out), exist_ok=True)
sdk.setup(avatar_path, tmp_out)
sdk.writer = writer
sdk.setup_Nd(N_d=num_f)

t0 = time.time()
padded = np.concatenate([np.zeros((CHUNKSIZE[0] * 640,), dtype=np.float32), audio], 0)
split_len = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
for i in range(0, len(padded), CHUNKSIZE[1] * 640):
    chunk = padded[i : i + split_len]
    if len(chunk) < split_len:
        chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
    sdk.run_chunk(chunk, CHUNKSIZE)
sdk.close()
elapsed = time.time() - t0

audio_secs = len(audio) / 16000
print(f"frames={writer.count}  elapsed={elapsed:.1f}s  audio={audio_secs:.1f}s")
print(f"fps={writer.count / elapsed:.1f}  realtime_factor={audio_secs / elapsed:.2f}x")
