#!/bin/bash
# run_local.sh — Run the evolution locally (no HuggingFace Space needed)
# Usage: paste your API keys below, then: bash run_local.sh

export PATH="$HOME/.local/bin:$PATH"

# ── Paste your keys here ──────────────────────────────────────────────
export GROQ_API_KEY=""
export GoogleAPIKey=""
export CEREBRAS_API_KEY=""
export SAMBANOVA_API_KEY=""
export OPENROUTER_API_KEY=""
# ─────────────────────────────────────────────────────────────────────

# Validate at least one key is set
if [ -z "$GROQ_API_KEY" ] && [ -z "$GoogleAPIKey" ] && [ -z "$CEREBRAS_API_KEY" ]; then
  echo "❌  Set at least one API key in run_local.sh before running"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p data

# Start a minimal HTTP server so MISSION_CONTROL.html can fetch data/ files
# (file:// URLs block fetch() due to browser CORS)
PORT=7860
python3.14 -c "
import http.server, os, threading
os.chdir('$SCRIPT_DIR')
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass          # silence request logs
    def do_GET(self):
        # always return fresh 200 (no 304 Not Modified)
        import pathlib
        p = pathlib.Path(self.path.split('?')[0].lstrip('/'))
        if p.is_file():
            data = p.read_bytes()
            ct = 'application/json' if str(p).endswith('.json') else 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', ct)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()
server = http.server.HTTPServer(('', $PORT), H)
print(f'  Dashboard → http://localhost:$PORT/MISSION_CONTROL.html')
server.serve_forever()
" &
SERVER_PID=$!

echo "🚀 Starting evolution locally — Gen 1"
echo "   Dashboard: http://localhost:$PORT/MISSION_CONTROL.html"
echo "   Logs:      tail -f generations/gen_1/generation.log"
echo ""

cd generations/gen_1
python3.14 main.py --generation 1

# Clean up server when evolution exits
kill $SERVER_PID 2>/dev/null
