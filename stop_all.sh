#!/usr/bin/env bash
# Stops everything run_all.sh starts: STT :5001, LLM :5004, TTS :5002,
# vid_gen :5003, UI :8000, Caddy :8443. Matches by port, not by parent
# process, so it works no matter how they were launched (foreground,
# nohup'd in the background, or left over from a terminal that died).
cd "$(dirname "$0")"

PORTS=(5001 5002 5003 5004 8000 8443)

stopped=0
for port in "${PORTS[@]}"; do
  pids=$(lsof -ti tcp:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "stopping :$port (pid $pids)"
    kill $pids 2>/dev/null
    stopped=1
  fi
done

if [ "$stopped" = 1 ]; then
  sleep 2
  for port in "${PORTS[@]}"; do
    pids=$(lsof -ti tcp:"$port" 2>/dev/null)
    if [ -n "$pids" ]; then
      echo "force-killing :$port (pid $pids)"
      kill -9 $pids 2>/dev/null
    fi
  done
  echo "All stopped."
else
  echo "Nothing running on the known ports."
fi
