"""Start a local dashboard server and open the dashboard in browser."""
from __future__ import annotations

import http.server
import os
import socketserver
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PORT = 8000

os.chdir(BASE_DIR)
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}/dashboard.html"
    print(f"Dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
