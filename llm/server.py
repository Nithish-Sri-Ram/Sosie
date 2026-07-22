"""Sosie LLM - Groq chat over HTTP (streaming API, remote).

POST /chat  {"text": "...", "persona": "elon"}  -> {"reply": "..."}
GET  /health
Runs on http://localhost:5004

Needs GROQ_API_KEY (see .env.example). Runs on Mac during dev - Groq itself is
remote, this is just a thin key-holding proxy so the browser never sees the key.
"""
import json
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast = low latency
DEFAULT_PERSONA = os.getenv(
    "SOSIE_PERSONA",
    "You are Sosie, a warm, concise conversational avatar. "
    "Reply in 1-2 short spoken sentences. No markdown, no emojis.",
)
PERSONAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "personas")


def system_prompt(persona_id):
    """personas/<id>/persona.json's system_prompt, falling back to .env's default."""
    path = os.path.join(PERSONAS_DIR, persona_id or "", "persona.json")
    try:
        with open(path) as f:
            return json.load(f)["system_prompt"]
    except (OSError, KeyError, json.JSONDecodeError):
        return DEFAULT_PERSONA


client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = Flask(__name__)
CORS(app)


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify(error="missing 'text'"), 400
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt(body.get("persona"))},
            {"role": "user", "content": text},
        ],
    )
    return jsonify(reply=resp.choices[0].message.content.strip())


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5004)))
