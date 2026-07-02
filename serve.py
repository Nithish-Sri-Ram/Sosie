"""Zero-dependency static server for the Sosie voice UI.

Serves index.html at http://localhost:8000 so the browser will grant mic
access (getUserMedia needs a localhost/secure origin - file:// won't do).
Start the STT (5001) and TTS (5002) servers first, then open this.
"""
import http.server
import os
import socketserver

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.getenv("PORT", 8000))

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Sosie UI -> http://localhost:{PORT}")
    httpd.serve_forever()
