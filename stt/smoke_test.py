"""STT smoke test — transcribe a wav file directly, no server, no browser.

    python smoke_test.py path/to/audio.wav

Prints the transcript + wall-clock latency so you know the raw model speed
before any of the network/browser plumbing is involved.
"""
import os
import sys
import time

from faster_whisper import WhisperModel

audio = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
if not os.path.exists(audio):
    sys.exit(f"no such audio file: {audio}\nusage: python smoke_test.py path/to/audio.wav")

t0 = time.time()
model = WhisperModel("base", device="cpu", compute_type="int8")
print(f"model loaded in {time.time() - t0:.1f}s")

t0 = time.time()
segments, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
text = " ".join(s.text.strip() for s in segments).strip()
print(f"transcribe in {time.time() - t0:.2f}s  (audio {info.duration:.1f}s)")
print("TEXT:", text)
