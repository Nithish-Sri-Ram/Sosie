"""Sosie LLM — Groq chat over HTTP (streaming API, remote).

POST /chat  {"text": "..."}  -> {"reply": "..."}
GET  /health
Runs on http://localhost:5004

Needs GROQ_API_KEY (see .env.example). Runs on Mac during dev — Groq itself is
remote, this is just a thin key-holding proxy so the browser never sees the key.
"""
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast = low latency
PERSONA = os.getenv(
    "SOSIE_PERSONA",
    "You are Sosie, a warm, concise conversational avatar. "
    "Reply in 1-2 short spoken sentences. No markdown, no emojis.",
)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = Flask(__name__)
CORS(app)


@app.post("/chat")
def chat():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="missing 'text'"), 400
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": text},
        ],
    )
    return jsonify(reply=resp.choices[0].message.content.strip())


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5004)))
