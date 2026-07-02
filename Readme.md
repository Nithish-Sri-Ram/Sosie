# Sosie

A real-time conversational avatar. You speak into a mic; a synthesized persona
replies back with voice and a lip-synced face, live.

```
mic ─▶ faster-whisper ─▶ Groq (LLM) ─▶ CosyVoice2-0.5B ─▶ MuseTalk ─▶ live A/V out
        STT, local        remote        TTS, local          lip sync, GPU
```

STT, LLM proxy, and TTS run locally on Mac (MPS) during dev; Groq is remote and
light. MuseTalk needs CUDA and runs on a rented L40/A100. Each layer is
smoke-tested on its own (text→audio, then audio→video) before being wired into
one loop — so latency is isolated per component, not debugged in a tangle.

## Layers

| Layer | Dir | Port | Endpoint | Device |
|-------|-----|------|----------|--------|
| STT — faster-whisper | `stt/` | 5001 | `WS /ws` (live) · `POST /transcribe` | CPU/int8 (no MPS in CTranslate2) |
| LLM — Groq | `llm/` | 5004 | `POST /chat` | remote (needs `GROQ_API_KEY`) |
| TTS — CosyVoice2-0.5B¹ | `tts/` | 5002 | `POST /tts` | MPS → CPU |
| vid_gen — MuseTalk | `vid_gen/` | 5003 | `POST /generate` | CUDA only (GPU box) |
| Voice UI | `index.html` | 8000 | served by `serve.py` | browser |

¹ Swappable with Chatterbox-Turbo — same `POST /tts` contract, drop-in.

Each layer has `server.py`, `requirements.txt`, and a standalone `smoke_test.py`.
Heavy layers (`tts/`, `vid_gen/`) also have a `SETUP.md` (clone repo + weights).

## Secrets

Only the LLM layer has a secret. Copy `llm/.env.example` → `llm/.env` and add
your **`GROQ_API_KEY`**. `.env` files are git-ignored; `.env.example` is tracked.

## Smoke-test each layer first (recommended order)

```bash
# STT
cd stt && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py sample.wav

# LLM  (add llm/.env first)
cd ../llm && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py "How are you?"

# TTS  (see tts/SETUP.md — clone repo + download model first)
cd ../tts && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py "Hello from Sosie."

# vid_gen — GPU box only, see vid_gen/SETUP.md
python smoke_test.py audio.wav
```

## Run the live loop (STT + LLM + TTS on Mac)

Four terminals:
```bash
cd stt && source .venv/bin/activate && python server.py   # :5001
cd llm && source .venv/bin/activate && python server.py   # :5004
cd tts && source .venv/bin/activate && python server.py   # :5002
python serve.py                                           # :8000
```
Open **http://localhost:8000**, click *Start talking*, speak, click *Stop* —
Sosie transcribes → Groq replies → speaks it back.

> Prefer one venv? `pip install -r requirements.txt` at the root installs
> STT + LLM + TTS together (vid_gen stays on the GPU box).

## MuseTalk GPU sizing (L40 or A100?)

**Neither is required.** Inference needs only ~8 GB VRAM — a 16–24 GB card
(T4/A10/RTX 4090) is enough to validate audio→video. Between the two, pick the
**L40** (ample, cheaper); save the **A100** for batching/fine-tuning. Full table
in `vid_gen/SETUP.md`.

## Status

- STT + LLM + TTS wired into `index.html` (browser loop works today).
- `vid_gen` (MuseTalk) is validated standalone (audio→video) on the GPU box and
  not yet wired into the browser loop — that's the final GPU-side integration.

---
*Team-1 · Sosie · built by Dev.*
