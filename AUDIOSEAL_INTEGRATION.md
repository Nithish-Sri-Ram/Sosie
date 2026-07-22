# AudioSeal Integration — Sosie AI Avatar

Meta's [AudioSeal](https://github.com/facebookresearch/audioseal) is a fast, localized audio watermarking system.
Sosie uses it to embed an invisible digital provenance signature into every AI-generated speech response, confirming that the audio came from the Sosie pipeline and not a human.

---

## Why AudioSeal?

| Property | AudioSeal | Traditional watermarks |
|---|---|---|
| Imperceptibility | Imperceptible to human ear | Often audible artefacts |
| Real-time speed | GPU-accelerated, <10 ms | Usually offline batch |
| Localized detection | Per-second localization | Whole-file detection |
| Open-source | Meta AI Research (MIT) | Usually proprietary |

---

## Repository layout

```
Sosie/
├── audioseal/               <- Meta's original repo (cloned subdir)
│   ├── src/audioseal/       <- installable Python package
│   └── ...
└── audioseal_wrapper/       <- Sosie's thin integration layer
    ├── __init__.py          <- public API exports
    ├── watermarker.py       <- SosieWatermarker class + helpers
    └── smoke_test.py        <- standalone end-to-end sanity check
```

---

## How it fits into the pipeline

```
CosyVoice TTS
     |
     |  audio tensor (float32, shape [C, N])
     v
SosieWatermarker.embed_watermark()   <- audioseal_wrapper/watermarker.py
     |
     |  watermarked tensor (same shape, inaudible delta added)
     v
tensor_to_wav_bytes()                <- converts to 16-bit PCM WAV in-memory
     |
     v
HTTP /tts response (audio/wav)  ->  MuseTalk lip-sync  ->  Browser
```

Detection (optional):

```
Incoming user audio (WAV bytes)
     |
wav_bytes_to_tensor()
     |
SosieWatermarker.detect_watermark()
     |
{is_watermarked: bool, score: float, message: list}
```

---

## Core API (audioseal_wrapper)

### SosieWatermarker class

Located in `audioseal_wrapper/watermarker.py`.

#### Constructor

```python
from audioseal_wrapper import get_watermarker
wm = get_watermarker()          # returns a singleton SosieWatermarker
```

Loads two AudioSeal pre-trained models:
- **Generator** (`audioseal_wm_16bits`) — embeds the watermark signal.
- **Detector** (`audioseal_detector_16bits`) — detects watermark and scores confidence.

Auto-selects `cuda` -> `cpu` depending on available hardware.

#### embed_watermark(audio_tensor, sample_rate, alpha, message)

| Arg | Type | Default | Description |
|---|---|---|---|
| `audio_tensor` | `torch.Tensor` | — | `(C, N)` or `(B, C, N)` float32 in `[-1, 1]` |
| `sample_rate` | `int` | `24000` | Sample rate (CosyVoice default) |
| `alpha` | `float` | `1.2` | Watermark strength; higher = more robust, slightly more audible |
| `message` | `torch.Tensor or None` | `None` | Optional 16-bit payload tensor |

Returns a `torch.Tensor` of the same shape.

How it works internally:

```python
watermark = generator.get_watermark(wav)      # delta signal, same shape
watermarked = wav + (alpha * watermark)       # additive blend
```

#### detect_watermark(audio_tensor, sample_rate)

Returns:

```python
{
  "is_watermarked": bool,   # score > 0.5
  "score": float,           # 0.0 to 1.0 confidence
  "message": list or None   # decoded 16-bit payload if present
}
```

---

### Helper utilities

#### tensor_to_wav_bytes(audio_tensor, sample_rate) -> io.BytesIO

Converts a float32 PyTorch tensor [-1, 1] to 16-bit PCM mono WAV in-memory buffer.
Used in `tts/server.py` to build the HTTP response.

#### wav_bytes_to_tensor(file_bytes_or_buf) -> (torch.Tensor, int)

Reads a WAV buffer/file -> float32 tensor `(1, 1, N)` + sample rate.
Used in `tts/server.py`'s `/detect_watermark` endpoint.

---

## TTS server changes (tts/server.py)

Three additions were made to the existing TTS server:

### 1. Watermarker import (graceful degradation)

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from audioseal_wrapper import get_watermarker, tensor_to_wav_bytes, wav_bytes_to_tensor
    watermarker = get_watermarker()
    print("AudioSeal Watermarker initialized in TTS service.")
except Exception as e:
    from audioseal_wrapper.watermarker import tensor_to_wav_bytes, wav_bytes_to_tensor
    watermarker = None
    print(f"Warning: AudioSeal not initialized ({e}). Serving raw TTS audio.")
```

The try/except ensures the TTS server still starts even if `audioseal` is not installed.

### 2. Watermark embedding in /tts

```python
if enable_wm and watermarker is not None:
    try:
        audio = watermarker.embed_watermark(audio, sample_rate=cosyvoice.sample_rate, alpha=1.2)
    except Exception as e:
        print(f"AudioSeal watermark embedding error: {e}")

buf = tensor_to_wav_bytes(audio, cosyvoice.sample_rate)
return send_file(buf, mimetype="audio/wav")
```

Callers can disable watermarking per-request: `{"text": "...", "watermark": false}`.

### 3. New /detect_watermark endpoint

```
POST /detect_watermark   multipart/form-data  field: audio (WAV file)
-> JSON: {is_watermarked, score, message}
```

### 4. /health now reports watermark status

```json
{"status": "ok", "watermarking_active": true}
```

---

## Setup

### 1. Install AudioSeal

The repo is already cloned at `Sosie/audioseal/`. Install it from there:

```bash
pip install -e audioseal/
```

Or from PyPI:

```bash
pip install audioseal
```

### 2. Environment variables

No secrets required. AudioSeal downloads pre-trained weights automatically via torch.hub on first run.

```bash
# Set automatically by watermarker.py on Windows to avoid MSVC compiler dependency
NO_TORCH_COMPILE=1
```

### 3. Run the smoke test

```bash
cd Sosie
python audioseal_wrapper/smoke_test.py
```

Expected output:

```
--- Running Sosie AudioSeal Smoke Test ---
[SosieWatermarker] Loading AudioSeal models on cpu...
[SosieWatermarker] AudioSeal loaded successfully.
Generated test sine wave audio shape: torch.Size([1, 48000])
Detection BEFORE watermarking: {'is_watermarked': False, 'score': ...}
Embedding AudioSeal watermark...
Watermarked audio shape: torch.Size([1, 48000])
Detection AFTER watermarking: {'is_watermarked': True, 'score': ...}
Detection after WAV buffer round-trip: {'is_watermarked': True, 'score': ...}

OK Sosie AudioSeal Integration Smoke Test PASSED SUCCESSFULLY!
```

---

## Graceful degradation

| AudioSeal installed? | Behaviour |
|---|---|
| Yes | Watermark embedded silently; /health shows `watermarking_active: true` |
| No | TTS still works; audio is un-watermarked; warning logged at startup |

---

## Files changed / created

| File | Change |
|---|---|
| `audioseal/` | Cloned from `facebookresearch/audioseal` |
| `audioseal_wrapper/__init__.py` | New — public API exports |
| `audioseal_wrapper/watermarker.py` | New — SosieWatermarker class + helper utilities |
| `audioseal_wrapper/smoke_test.py` | New — end-to-end sanity check |
| `tts/server.py` | Modified — watermark embed on /tts, new /detect_watermark endpoint |
| `Readme.md` | Updated — AudioSeal section added |
| `AUDIOSEAL_INTEGRATION.md` | New — this document |

---

*AudioSeal paper: AudioSeal: Proactive Detection of Voice Cloning with Localized Watermarking — Roman Sanchez et al., Meta AI (2024).*
