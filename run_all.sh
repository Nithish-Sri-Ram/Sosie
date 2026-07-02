#!/usr/bin/env bash
# Starts the whole voice loop: STT :5001, LLM :5004, TTS :5002, UI :8000.
# Logs land in ./logs/. Ctrl-C stops everything.
set -e
cd "$(dirname "$0")"
mkdir -p logs

trap 'kill 0' EXIT INT TERM

python stt/server.py > logs/stt.log 2>&1 & echo "STT  http://localhost:5001  (logs/stt.log)"
python llm/server.py > logs/llm.log 2>&1 & echo "LLM  http://localhost:5004  (logs/llm.log)"
python tts/server.py > logs/tts.log 2>&1 & echo "TTS  http://localhost:5002  (logs/tts.log)"
python serve.py      > logs/ui.log  2>&1 & echo "UI   http://localhost:8000  (logs/ui.log)"
# vid_gen (MuseTalk) has its own venv - only start it if that env was set up
if [ -x vid_gen/.venv/bin/python ]; then
  vid_gen/.venv/bin/python vid_gen/server.py > logs/vid.log 2>&1 & echo "VID  http://localhost:5003  (logs/vid.log)"
fi
if command -v caddy > /dev/null; then
  caddy run --config Caddyfile > logs/caddy.log 2>&1 & echo "HTTPS https://<this-ip>:8443  (logs/caddy.log)"
fi

echo
echo "Waiting for TTS to load CosyVoice (first start takes ~1 min)..."
until curl -sf http://localhost:5002/health > /dev/null 2>&1; do sleep 2; done
echo "All up."
echo "  local / ssh tunnel:  http://localhost:8000"
echo "  over the internet:   https://$(curl -s4 --max-time 3 icanhazip.com 2>/dev/null || echo '<public-ip>'):8443  (self-signed cert - click through the warning)"
wait
