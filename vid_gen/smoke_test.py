"""MuseTalk smoke test — audio -> lip-synced video, standalone (GPU box).

    python smoke_test.py path/to/audio.wav

Runs MuseTalk's realtime inference directly and times it so you get the raw
audio->video latency / FPS before wiring vid_gen into the pipeline. CUDA only;
this will NOT run on the Mac.
"""
import os
import subprocess
import sys
import time

MUSETALK_DIR = os.getenv("MUSETALK_DIR", os.path.join(os.path.dirname(__file__), "MuseTalk"))
CONFIG = os.getenv("MUSETALK_CONFIG", "configs/inference/realtime.yaml")
audio = sys.argv[1] if len(sys.argv) > 1 else "data/audio/sample.wav"

t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-m", "scripts.realtime_inference",
     "--inference_config", CONFIG, "--audio_path", audio],
    cwd=MUSETALK_DIR,
)
print(f"\nexit={proc.returncode}  elapsed={time.time() - t0:.1f}s")
