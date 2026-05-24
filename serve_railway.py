"""
serve_railway.py — Combined HTTP server + evolution launcher for Railway.

Starts two threads:
  1. HTTP server on $PORT  (serves MISSION_CONTROL.html + data/ JSON files)
  2. Evolution runner      (runs gen_1/main.py, which chains into gen_2, gen_3, ...)

The dashboard auto-polls data/heartbeat.json every 3 seconds.
Set GROQ_API_KEY as a Railway environment variable — never hardcode it here.
"""
import http.server
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("railway")

# Railway injects PORT; fall back to 8080 for local testing
PORT = int(os.environ.get("PORT", 8080))
ROOT = Path(__file__).resolve().parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files quietly — only log errors."""
    def log_message(self, fmt, *args):
        if args[1] not in ("200", "304"):
            super().log_message(fmt, *args)


def run_http_server():
    os.chdir(ROOT)
    server = http.server.HTTPServer(("0.0.0.0", PORT), QuietHandler)
    logger.info(f"Dashboard → http://0.0.0.0:{PORT}/MISSION_CONTROL.html")
    server.serve_forever()


def run_evolution():
    """
    Start Gen 1, then let each generation self-launch the next.
    Restarts from Gen 1 if the lineage ends (all generations complete/fail).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — evolution cannot start. "
                     "Add it as a Railway environment variable.")
        return

    env = os.environ.copy()
    gen1_main = ROOT / "generations" / "gen_1" / "main.py"

    while True:
        if not gen1_main.exists():
            logger.error(f"Gen 1 main.py not found at {gen1_main}")
            time.sleep(60)
            continue

        logger.info("=== Starting evolution from Generation 1 ===")

        # Clean previous results so each Railway run is fresh
        results_dir = ROOT / "generations" / "gen_1" / "results"
        if results_dir.exists():
            for f in results_dir.glob("*.json"):
                f.unlink()

        log_path = ROOT / "generations" / "gen_1" / "generation.log"
        proc = subprocess.run(
            [sys.executable, str(gen1_main), "--generation", "1"],
            cwd=ROOT / "generations" / "gen_1",
            env=env,
        )

        exit_code = proc.returncode
        logger.info(f"Gen 1 process exited with code {exit_code}")

        # Wait before restarting to avoid hammering the API on crash loops
        logger.info("Waiting 300s before restarting evolution lineage...")
        time.sleep(300)


if __name__ == "__main__":
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=False)
    http_thread.start()
    logger.info(f"HTTP server started on port {PORT}")

    # Small delay so server is ready before evolution logs start
    time.sleep(2)

    # Run evolution in foreground (keeps the process alive)
    run_evolution()
