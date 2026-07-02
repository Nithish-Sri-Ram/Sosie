# vid_gen — MuseTalk (audio-driven lip sync)

**CUDA / NVIDIA only.** Does not run on this Mac. Serves `POST /generate` -> mp4 on **:5003**.

## GPU sizing — do you need an L40 or A100?
**No — neither is required for inference.** MuseTalk inference is light:

| GPU | VRAM | Verdict for MuseTalk inference |
|-----|------|--------------------------------|
| T4 / RTX 3090 / A10 | 16–24 GB | ✅ enough; real-time-ish |
| **L40 / L40S** | 48 GB | ✅ plenty, comfortable headroom |
| **A100** | 40/80 GB | ✅ works, but overkill for inference |

Inference needs only **~8 GB VRAM**. Pick an **L40** if choosing between the two
(cheaper, ample); reserve **A100** for when you're batching many streams or
fine-tuning. A 16–24 GB card (T4/A10/4090) is the cheapest thing that works if
you just want to validate the audio->video layer.

## Setup (on the GPU box)
```bash
python -m venv .venv && source .venv/bin/activate
git clone https://github.com/TMElyralab/MuseTalk.git   # server.py expects ./MuseTalk

# torch matching your CUDA, then deps
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r MuseTalk/requirements.txt
# OpenMMLab for DWPose (version-sensitive — follow MuseTalk README exactly):
pip install --no-cache-dir -U openmim && mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

# weights (sd-vae, whisper, dwpose, musetalk) — script in the repo
cd MuseTalk && sh ./download_weights.sh && cd ..
```

## Smoke test first (independent audio -> video)
```bash
python smoke_test.py path/to/audio.wav   # times raw audio->video before wiring
```
`server.py` wraps `scripts.realtime_inference`. You must point `MUSETALK_CONFIG`
at a prepared avatar config and confirm the output naming matches what
`server.py` looks for — adjust that one line to your config.
