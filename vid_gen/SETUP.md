# vid_gen - Ditto (full-face audio-driven avatar)

**CUDA / NVIDIA only.** Does not run on this Mac. Serves `POST /generate` and
`POST /generate_stream` -> mp4 on **:5003**.

Ditto (`antgroup/ditto-talkinghead`) generates full head motion and expression
from a single reference image + audio - not just lip-sync onto a fixed driving
video, which is what the old MuseTalk-based version did.

## GPU sizing

Measured on an A30 (24 GB): ~2.6 GB VRAM peak, ~0.85x real-time with the
prebuilt TensorRT engines - workable but not comfortable once STT+TTS share
the card. On an A100 (80 GB): ~1.3x real-time, comfortable headroom. Any
Ampere-or-newer card (A30/A100/RTX 30xx+) can use the prebuilt
`ditto_trt_Ampere_Plus` engines directly; older GPUs need to rebuild the
TensorRT engines from ONNX first (see the upstream repo's `cvt_onnx_to_trt.py`).

## Setup (on the GPU box)

```bash
cd vid_gen
git clone https://github.com/antgroup/ditto-talkinghead.git ditto/repo
cd ditto/repo

python -m venv .venv && source .venv/bin/activate

# torch matching your CUDA build first
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# tensorrt needs --no-build-isolation (its setup.py shells out to pip), and
# cuda-python must stay on a 12.x release - 13.x dropped the flat
# `from cuda import cuda, cudart, nvrtc` namespace this repo imports
pip install tensorrt==8.6.1 --no-build-isolation --extra-index-url https://pypi.nvidia.com
pip install cuda-python==12.6.0

# the rest of Ditto's own deps, then the Flask server's
pip install -r ../../requirements.txt   # flask, librosa, imageio-ffmpeg, etc.

# checkpoints (~13 GB: onnx, TensorRT engines, pytorch weights) - see the
# upstream README for the download link, unpack into ditto/repo/checkpoints/
```

`ffmpeg` doesn't need to be on `PATH` - `server.py` resolves the bundled
binary via `imageio_ffmpeg.get_ffmpeg_exe()`.

## Smoke test first (independent audio -> video)

```bash
python smoke_test.py path/to/audio.wav   # times raw audio->video before wiring
```

## Avatars

`server.py` reads `personas/<id>/avatar.jpg` per request (no pre-baking
needed - registration is cheap, ~0.05-0.3s). Run with
`vid_gen/ditto/repo/.venv/bin/python vid_gen/server.py` so it picks up
Ditto's own venv rather than a shared one.
