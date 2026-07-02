"""TTS smoke test — text -> out.wav directly, no server, no browser.

    python smoke_test.py "Hello from Sosie."

Confirms CosyVoice2 loads, imports resolve, and prints synthesis latency +
real-time factor. Needs the same setup as server.py (cloned repo, weights,
assets/prompt.wav). Run this BEFORE wiring TTS into the pipeline.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CosyVoice"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "CosyVoice", "third_party", "Matcha-TTS")
)

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is Sosie speaking."

t0 = time.time()
cosyvoice = CosyVoice2(
    "CosyVoice/pretrained_models/CosyVoice2-0.5B",
    load_jit=False, load_trt=False, fp16=False,
)
prompt_16k = load_wav("assets/prompt.wav", 16000)
print(f"model loaded in {time.time() - t0:.1f}s")

t0 = time.time()
chunks = [o["tts_speech"] for o in
          cosyvoice.inference_zero_shot(text, "Hello, this is a reference voice sample.",
                                        prompt_16k, stream=False)]
audio = torch.cat(chunks, dim=1)
dur = audio.shape[1] / cosyvoice.sample_rate
elapsed = time.time() - t0
torchaudio.save("out.wav", audio, cosyvoice.sample_rate)
print(f"synth in {elapsed:.2f}s  audio {dur:.1f}s  RTF {elapsed / dur:.2f}  -> out.wav")
