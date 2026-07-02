"""LLM smoke test - one Groq round-trip, no server, no browser.

    python smoke_test.py "How's the weather on Mars?"

Confirms GROQ_API_KEY works and prints reply + round-trip latency.
"""
import os
import sys
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."

t0 = time.time()
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are Sosie. Reply in one short spoken sentence."},
        {"role": "user", "content": prompt},
    ],
)
print(f"reply in {time.time() - t0:.2f}s ({model})")
print("REPLY:", resp.choices[0].message.content.strip())
