# AudioSeal integration

Meta's [AudioSeal](https://github.com/facebookresearch/audioseal) is a fast,
localized audio watermarking model. Sosie uses it to embed an inaudible
provenance signal into every AI-generated speech response, so the audio can
later be confirmed as synthetic rather than a real human recording.

`audioseal` is a normal pip dependency (see `tts/requirements.txt`) - no
vendored/cloned copy in this repo.

## Pipeline

```
CosyVoice TTS
  -> audio tensor (float32, shape [C, N])
  -> SosieWatermarker.embed_watermark()   (audioseal_wrapper/watermarker.py)
  -> watermarked tensor, same shape, inaudible delta added
  -> tensor_to_wav_bytes()                -> 16-bit PCM WAV
  -> HTTP /tts response (audio/wav) -> vid_gen (lip sync) -> browser
```

Detection (optional, on any WAV):
```
wav bytes -> wav_bytes_to_tensor() -> SosieWatermarker.detect_watermark()
  -> {is_watermarked, score, message}
```

## audioseal_wrapper/watermarker.py

`get_watermarker()` returns a process-wide singleton `SosieWatermarker` that
loads two pretrained AudioSeal models on construction (generator +
detector), auto-selecting `cuda` over `cpu` when available.

**`embed_watermark(audio_tensor, sample_rate=24000, alpha=1.2, message=None)`**
Accepts `(C, N)` or `(B, C, N)` float32 in `[-1, 1]`, returns a tensor of the
same shape. `alpha` trades watermark strength for audibility (higher =
more robust to re-encoding, slightly more audible). Internally:
```python
watermark = generator.get_watermark(wav, message=message)
watermarked = wav + (alpha * watermark)
```

**`detect_watermark(audio_tensor, sample_rate=16000)`**
Returns `{"is_watermarked": bool, "score": float, "message": list or None}`
- `score` is 0.0-1.0 confidence, `is_watermarked` is `score > 0.5`.

**Helpers**
- `tensor_to_wav_bytes(tensor, sample_rate) -> io.BytesIO` - float32 [-1,1] to
  16-bit PCM mono WAV.
- `wav_bytes_to_tensor(buf) -> (tensor, sample_rate)` - inverse, returns
  `(1, 1, N)` float32.

## tts/server.py changes

- Import is wrapped in try/except so the TTS server still starts if
  `audioseal` isn't installed - `watermarker` stays `None` and `/tts` serves
  unwatermarked audio with a warning logged at startup.
- `/tts` embeds the watermark (`alpha=1.2`) after CosyVoice synthesis unless
  the request body has `"watermark": false`.
- `POST /detect_watermark` - multipart form field `audio` (a WAV file) ->
  JSON detection result.
- `GET /health` now includes `"watermarking_active": true|false`.

## Setup

```bash
pip install audioseal
```

No secrets needed. Pretrained weights download automatically via
`torch.hub` on first use, cached under `~/.cache/audioseal/`. If that
download stalls on a slow/throttled connection, the loader falls back to
whatever's already at the expected cache path - fetching the two files
directly (`generator_base.pth`, `detector_base.pth` from
`huggingface.co/facebook/audioseal`) and placing them there works too.

Smoke test:
```bash
python audioseal_wrapper/smoke_test.py
```
Expected: detection before watermarking reports `is_watermarked: False`,
after embedding reports `True` with `score: 1.0`, and that survives a
round-trip through the WAV buffer helpers.

## Graceful degradation

If `audioseal` isn't installed, `/tts` still serves audio (just
unwatermarked) and `/detect_watermark` returns a 503. Nothing else in the
pipeline depends on it.
