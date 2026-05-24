"""Simple HTTP server for MISSION_CONTROL dashboard."""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet logging — only log non-200 responses
        if args[1] != '200':
            super().log_message(format, *args)

server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
print(f"MISSION CONTROL dashboard at: http://localhost:{PORT}/MISSION_CONTROL.html")
print("Press Ctrl+C to stop.")
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
