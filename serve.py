"""Zero-dependency static server for the Sosie voice UI.

Serves index.html at http://localhost:8000 so the browser will grant mic
access (getUserMedia needs a localhost/secure origin - file:// won't do).
Start the STT (5001) and TTS (5002) servers first, then open this.
"""
import http.server
import os
import socketserver


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """The UI iterates fast - never let browsers serve a stale index.html."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()


os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.getenv("PORT", 8000))

with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
    print(f"Sosie UI -> http://localhost:{PORT}")
    httpd.serve_forever()
