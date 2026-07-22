# Sosie

A real-time conversational avatar. You speak into a mic; a synthesized persona
replies back with voice and a lip-synced face, live.

```
mic -> faster-whisper -> Groq (LLM) -> CosyVoice2-0.5B -> MuseTalk -> live A/V out
        STT, local        remote        TTS, local          lip sync, GPU
```

STT, LLM proxy, and TTS run locally on Mac (MPS) during dev; Groq is remote and
light. MuseTalk needs CUDA and runs on a rented L40/A100. Each layer is
smoke-tested on its own (text->audio, then audio->video) before being wired into
one loop - so latency is isolated per component, not debugged in a tangle.

## Layers

| Layer | Dir | Port | Endpoint | Device |
|-------|-----|------|----------|--------|
| STT - faster-whisper | `stt/` | 5001 | `WS /ws` (live) - `POST /transcribe` | CPU/int8 (no MPS in CTranslate2) |
| LLM - Groq | `llm/` | 5004 | `POST /chat` | remote (needs `GROQ_API_KEY`) |
| TTS - CosyVoice2-0.5B[1] | `tts/` | 5002 | `POST /tts` | MPS -> CPU |
| AudioSeal watermarker | `audioseal_wrapper/` | — | embedded in TTS `/tts` + `POST /detect_watermark` | CPU / CUDA |
| vid_gen - MuseTalk | `vid_gen/` | 5003 | `POST /generate` | CUDA only (GPU box) |
| Voice UI | `index.html` | 8000 | served by `serve.py` | browser |

[1] Swappable with Chatterbox-Turbo - same `POST /tts` contract, drop-in.

Each layer has `server.py`, `requirements.txt`, and a standalone `smoke_test.py`.

## AudioSeal — AI Audio Watermarking

We integrated Meta's [AudioSeal](https://github.com/facebookresearch/audioseal) library
to embed an invisible provenance watermark into every TTS-generated audio clip.

**What was done:**

1. Cloned `facebookresearch/audioseal` into `Sosie/audioseal/` (pip-installable via `pip install -e audioseal/`).
2. Created `audioseal_wrapper/` — a thin Sosie-specific layer with:
   - `watermarker.py` — `SosieWatermarker` class (embed + detect), `tensor_to_wav_bytes`, `wav_bytes_to_tensor` helpers.
   - `__init__.py` — exposes `get_watermarker`, `tensor_to_wav_bytes`, `wav_bytes_to_tensor`.
   - `smoke_test.py` — end-to-end test: generates a sine wave, embeds watermark, detects it, round-trips through WAV buffer.
3. Modified `tts/server.py`:
   - Watermark is automatically embedded by `embed_watermark(alpha=1.2)` after CosyVoice synthesis.
   - Callers may opt out per-request: `{"text": "...", "watermark": false}`.
   - New `POST /detect_watermark` endpoint (multipart WAV upload → JSON score).
   - `GET /health` now reports `watermarking_active: true/false`.
   - Graceful fallback: TTS still serves audio if `audioseal` is not installed.
4. Set `NO_TORCH_COMPILE=1` in the wrapper to prevent Windows MSVC compile errors.

Full details: see [`AUDIOSEAL_INTEGRATION.md`](AUDIOSEAL_INTEGRATION.md).

Smoke-test the watermarker standalone:
```bash
python audioseal_wrapper/smoke_test.py
```
Heavy layers (`tts/`, `vid_gen/`) also have a `SETUP.md` (clone repo + weights).

## Secrets

Only the LLM layer has a secret. Copy `llm/.env.example` -> `llm/.env` and add
your **`GROQ_API_KEY`**. `.env` files are git-ignored; `.env.example` is tracked.

## Smoke-test each layer first (recommended order)

```bash
# STT
cd stt && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py sample.wav

# LLM  (add llm/.env first)
cd ../llm && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py "How are you?"

# TTS  (see tts/SETUP.md - clone repo + download model first)
cd ../tts && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python smoke_test.py "Hello from Sosie."

# vid_gen - GPU box only, see vid_gen/SETUP.md
python smoke_test.py audio.wav
```

## Run the live loop (STT + LLM + TTS)

One command (starts all four, logs in `./logs/`):
```bash
./run_all.sh
```
Or four terminals:
```bash
cd stt && source .venv/bin/activate && python server.py   # :5001
cd llm && source .venv/bin/activate && python server.py   # :5004
cd tts && source .venv/bin/activate && python server.py   # :5002
python serve.py                                           # :8000
```
Open **http://localhost:8000**, click *Start talking*, speak, click *Stop* -
Sosie transcribes -> Groq replies -> speaks it back in the cloned voice.

The persona (default: Elon Musk, see `llm/.env`) is set via `SOSIE_PERSONA`;
the cloned voice comes from `tts/.env` (`PROMPT_WAV` = reference clip,
`PROMPT_TEXT` = exact words spoken in it). Reference assets live in `assets/`.

### Open it from anywhere (HTTPS reverse proxy)

Mic access (`getUserMedia`) needs a secure origin, so plain `http://<ip>:8000`
won't work from outside. Two options:

**1. Caddy HTTPS proxy (started by `run_all.sh`)** - one origin fronts the UI
and all three backends (see `Caddyfile`): open **`https://<public-ip>:8443`**,
click through the self-signed-cert warning once, and talk. Regenerate the cert
for a new box (put its IP in the SAN):

```bash
mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=sosie" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<public-ip>"
```

Caddy itself is a single static binary:
`curl -sL https://github.com/caddyserver/caddy/releases/download/v2.10.2/caddy_2.10.2_linux_amd64.tar.gz | tar xz caddy && mv caddy /usr/local/bin/`

**2. SSH tunnel** - everything looks like localhost, no cert warning:
`ssh -L 8000:localhost:8000 -L 5001:localhost:5001 -L 5002:localhost:5002 -L 5004:localhost:5004 user@gpu-box`
then open http://localhost:8000.

> Prefer one venv? `pip install -r requirements.txt` at the root installs
> STT + LLM + TTS together (vid_gen stays on the GPU box).

## MuseTalk GPU sizing (L40 or A100?)

**Neither is required.** Inference needs only ~8 GB VRAM - a 16-24 GB card
(T4/A10/RTX 4090) is enough to validate audio->video. Between the two, pick the
**L40** (ample, cheaper); save the **A100** for batching/fine-tuning. Full table
in `vid_gen/SETUP.md`.

## Status / What we built

### Core pipeline (verified end-to-end on A30 GPU box)

- **STT** (`stt/`, :5001) — faster-whisper + Silero VAD for server-side turn detection.
  Speak, pause; the turn is detected automatically and transcription is sent downstream.
- **LLM** (`llm/`, :5004) — Groq-hosted Llama-3.1-8b-instant; persona set via `SOSIE_PERSONA` env var (default: Elon Musk).
- **TTS** (`tts/`, :5002) — CosyVoice2-0.5B zero-shot voice clone; reference clip in `assets/elon_musk_sample.wav`.
- **vid_gen** (`vid_gen/`, :5003) — MuseTalk V1.5 lip-sync; streams fragmented mp4 while frames are still rendering; UI plays live via MediaSource API with fallback to whole-file mp4, then voice-only.
- **Voice UI** (`index.html`, :8000) — single-page mic UI; served by `serve.py`.
- **HTTPS reverse proxy** (`Caddyfile`, :8443) — mic works from any browser, including JarvisLabs port proxy (`/proxy/8000/`).
- **Orchestration** (`run_all.sh`) — launches all four services, logs to `./logs/`.

### AudioSeal AI watermarking (this session)

- Cloned `facebookresearch/audioseal` → `Sosie/audioseal/`
- Created `audioseal_wrapper/` package:
  - `watermarker.py` — `SosieWatermarker` class (embed + detect), WAV/tensor helpers
  - `__init__.py` — public API exports
  - `smoke_test.py` — standalone sanity test (embed → detect → WAV round-trip)
- Modified `tts/server.py`:
  - Watermark embedded after every CosyVoice synthesis (alpha=1.2, imperceptible)
  - Per-request opt-out: `{"watermark": false}`
  - New `POST /detect_watermark` endpoint
  - `/health` reports `watermarking_active` flag
  - Graceful degradation if `audioseal` not installed
- Full integration doc: [`AUDIOSEAL_INTEGRATION.md`](AUDIOSEAL_INTEGRATION.md)

---
*Team-1 - Sosie.*

