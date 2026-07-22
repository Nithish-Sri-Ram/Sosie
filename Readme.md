# Sosie

A real-time conversational avatar. You speak into a mic; a synthesized persona
replies back with voice and a full-face avatar, live.

```
mic -> faster-whisper -> Groq (LLM) -> CosyVoice2-0.5B -> Ditto -> live A/V out
        STT, local        remote        TTS, local        avatar, GPU
```

STT, LLM proxy, and TTS run locally on Mac (MPS) during dev; Groq is remote and
light. Ditto needs CUDA and runs on a rented GPU box (A30 or better). Each
layer is smoke-tested on its own (text->audio, then audio->video) before being
wired into one loop - so latency is isolated per component, not debugged in a
tangle.

## Layers

| Layer | Dir | Port | Endpoint | Device |
|-------|-----|------|----------|--------|
| STT - faster-whisper | `stt/` | 5001 | `WS /ws` (live) - `POST /transcribe` | CPU/int8 (no MPS in CTranslate2) |
| LLM - Groq | `llm/` | 5004 | `POST /chat` | remote (needs `GROQ_API_KEY`) |
| TTS - CosyVoice2-0.5B[1] | `tts/` | 5002 | `POST /tts` | MPS -> CPU |
| AudioSeal watermarker | `audioseal_wrapper/` | - | embedded in TTS `/tts` + `POST /detect_watermark` | CPU / CUDA |
| vid_gen - Ditto | `vid_gen/` | 5003 | `POST /generate_stream` (`/generate`, `/idle`, `/personas`) | CUDA only (GPU box) |
| Voice UI | `index.html` | 8000 | served by `serve.py` | browser |

[1] Swappable with Chatterbox-Turbo - same `POST /tts` contract, drop-in.

Each layer has `server.py`, `requirements.txt`, and a standalone `smoke_test.py`.

Personas are config-driven: drop a folder in `personas/<id>/` (`persona.json`
+ `avatar.jpg` + `voice.wav`) and it shows up in the UI's picker automatically,
no code changes - LLM, TTS, and vid_gen all resolve persona assets off the
same directory per-request.

## AudioSeal audio watermarking

Every TTS response is watermarked using Meta's [AudioSeal](https://github.com/facebookresearch/audioseal),
so generated speech carries an inaudible signature identifying it as
synthetic. Opt out per request with `{"text": "...", "watermark": false}`.
`POST /detect_watermark` (multipart WAV upload) checks any clip and returns a
confidence score - useful for telling real audio apart from Sosie output.

Full details: [`AUDIOSEAL_INTEGRATION.md`](AUDIOSEAL_INTEGRATION.md).

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
cd ../vid_gen && ditto/repo/.venv/bin/python smoke_test.py audio.wav
```

## Run the live loop

One command (starts everything, logs in `./logs/`):
```bash
./run_all.sh
```
Stop everything with `./stop_all.sh` - it kills by port, so it works even if
`run_all.sh`'s own terminal already closed or died.

Or run each service by hand:
```bash
cd stt && source .venv/bin/activate && python server.py             # :5001
cd llm && source .venv/bin/activate && python server.py             # :5004
cd tts && source .venv/bin/activate && python server.py             # :5002
cd vid_gen && ditto/repo/.venv/bin/python server.py                 # :5003
python serve.py                                                     # :8000
```
Open **http://localhost:8000**, pick a persona, click *Start* and talk -
Sosie transcribes -> Groq replies -> speaks it back in the cloned voice with a
live avatar.

Voice cloning reference is per-persona: `personas/<id>/voice.wav` +
`persona.json`'s `voice_text` (the exact words spoken in the clip). The LLM
persona (system prompt) comes from the same `persona.json`.

### Open it from anywhere (HTTPS reverse proxy)

Mic access (`getUserMedia`) needs a secure origin, so plain `http://<ip>:8000`
won't work from outside. Two options:

**1. Caddy HTTPS proxy (started by `run_all.sh`)** - one origin fronts the UI
and all four backends (see `Caddyfile`): open **`https://<public-ip>:8443`**,
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
`ssh -L 8000:localhost:8000 -L 5001:localhost:5001 -L 5002:localhost:5002 -L 5004:localhost:5004 -L 5003:localhost:5003 user@gpu-box`
then open http://localhost:8000.

> Prefer one venv? `pip install -r requirements.txt` at the root installs
> STT + LLM + TTS together (vid_gen needs its own venv on the GPU box, see
> `vid_gen/SETUP.md`).

## GPU sizing for vid_gen (Ditto)

Needs an Ampere-class GPU or newer to use the prebuilt TensorRT engines
directly. Measured on an A30 (24 GB): ~2.6 GB VRAM, ~0.85x real-time -
workable but tight once STT+TTS share the card. On an A100 (80 GB): ~1.3x
real-time, comfortable headroom. Full detail in `vid_gen/SETUP.md`.

## Status

Verified end-to-end on a GPU box (STT -> LLM -> TTS -> vid_gen -> UI, full
round trip):

- **STT** (`stt/`, :5001) - faster-whisper + Silero VAD for server-side turn
  detection. Speak, pause; the turn is detected automatically.
- **LLM** (`llm/`, :5004) - Groq-hosted Llama-3.1-8b-instant; persona's system
  prompt read from `personas/<id>/persona.json`.
- **TTS** (`tts/`, :5002) - CosyVoice2-0.5B zero-shot voice clone from
  `personas/<id>/voice.wav`; every response watermarked via AudioSeal.
- **vid_gen** (`vid_gen/`, :5003) - Ditto full-face avatar from
  `personas/<id>/avatar.jpg`; streams fragmented mp4 while frames are still
  rendering, UI plays live via MediaSource with fallback to whole-file mp4,
  then voice-only.
- **Voice UI** (`index.html`, :8000) - persona picker + call UI, served by
  `serve.py`.
- **HTTPS reverse proxy** (`Caddyfile`, :8443) - mic works from any browser,
  including a JarvisLabs-style port proxy (`/proxy/8000/`).
- **Orchestration** (`run_all.sh`) - launches everything, logs to `./logs/`.

---
*Team-1 - Sosie.*
