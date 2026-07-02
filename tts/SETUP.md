# TTS — CosyVoice2-0.5B

Prefers Apple **MPS**, falls back to CPU. Serves `POST /tts {text}` -> wav on **:5002**.

## Setup (you run this in your venv)
```bash
python -m venv .venv && source .venv/bin/activate

# 1. Clone CosyVoice INTO this folder (server.py expects ./CosyVoice)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# 2. Deps: our server bits + CosyVoice's full stack
pip install -r requirements.txt
pip install -r CosyVoice/requirements.txt

# 3. Download the model
python -c "from modelscope import snapshot_download; \
  snapshot_download('iic/CosyVoice2-0.5B', local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')"

# 4. Reference voice for zero-shot cloning: a 3–10s clean clip
mkdir -p assets && cp /path/to/your_voice.wav assets/prompt.wav
# set PROMPT_TEXT to the exact words spoken in that clip (default is a placeholder)
```

### Caveats
- **Text normalization** (`pynini` / `WeTextProcessing`) often needs conda/OpenFst:
  `conda install -y -c conda-forge pynini==2.1.5 && pip install WeTextProcessing`.
  CosyVoice runs without it but numbers/dates read less cleanly.
- MPS coverage in CosyVoice is partial; if you hit an unsupported op, run CPU
  (`PYTORCH_ENABLE_MPS_FALLBACK=1` helps).

## Smoke test first (do this before wiring anything)
```bash
python smoke_test.py "Hello from Sosie."   # -> out.wav, prints latency + RTF
```
Then run the server:
```bash
python server.py
curl -X POST localhost:5002/tts -H 'Content-Type: application/json' \
     -d '{"text":"testing one two three"}' --output reply.wav
```
