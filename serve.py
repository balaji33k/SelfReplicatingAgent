"""
serve.py — Combined HTTP server + evolution launcher for HuggingFace Spaces.

Starts two threads:
  1. HTTP server on $PORT  (serves MISSION_CONTROL.html + data/ JSON files)
  2. Evolution runner      (runs gen_1/main.py, which chains into gen_2, gen_3, ...)

The dashboard auto-polls data/heartbeat.json every 3 seconds.
Set GROQ_API_KEY as an environment variable — never hardcode it here.
"""
import http.server
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("server")

# HuggingFace uses 7860, Railway uses $PORT, fallback 8080
PORT = int(os.environ.get("PORT", 8080))
ROOT = Path(__file__).resolve().parent

# ── Global state shared between HTTP handler and evolution loop ───────────────
_restart_event = threading.Event()   # set → evolution loop kills current run & restarts
_current_proc: subprocess.Popen = None  # current gen_1 subprocess (so we can kill it)
_proc_lock = threading.Lock()

# Known Groq free-tier models shown in the dashboard selector
GROQ_FREE_MODELS = [
    {"id": "meta-llama/llama-4-scout-17b-16e-instruct",  "name": "Llama 4 Scout 17B",       "tpd": "500k", "tpm": "30k"},
    {"id": "llama-3.1-8b-instant",                        "name": "Llama 3.1 8B Instant",    "tpd": "500k", "tpm": "20k"},
    {"id": "llama-3.3-70b-versatile",                     "name": "Llama 3.3 70B Versatile", "tpd": "100k", "tpm": "12k"},
    {"id": "deepseek-r1-distill-llama-70b",               "name": "DeepSeek R1 Llama 70B",   "tpd": "500k", "tpm": "30k"},
    {"id": "deepseek-r1-distill-qwen-32b",                "name": "DeepSeek R1 Qwen 32B",    "tpd": "500k", "tpm": "30k"},
    {"id": "qwen-qwq-32b",                                "name": "Qwen QwQ 32B",            "tpd": "500k", "tpm": "30k"},
]


AGENT_FILES = [
    "main.py", "config.py", "pipeline.py", "contracts.py",
    "agent_base.py", "llm_client.py", "task_manager.py",
    "analysis.py", "evolution_engine.py", "spawner.py",
    "agent_analyst.py", "agent_architect.py", "agent_coder.py",
    "agent_critic.py", "agent_reviser.py", "agent_test_writer.py",
    "agent_debugger.py", "agent_clone_inspector.py",
    "telemetry.py", "evolution_tracker.py",
]


def _build_code_browser() -> str:
    """Generate the /code HTML page listing all generation files."""
    gens_dir = ROOT / "generations"
    gen_dirs = sorted(gens_dir.glob("gen_*"), key=lambda p: int(p.name.split("_")[1]))

    sections = []
    for gd in gen_dirs:
        gen_num = gd.name  # "gen_1", "gen_2", ...
        # Agent source files
        agent_links = []
        for fname in AGENT_FILES:
            fp = gd / fname
            if fp.exists():
                url = f"/generations/{gen_num}/{fname}"
                agent_links.append(
                    f'<a href="{url}" target="_blank" style="display:inline-block;'
                    f'padding:4px 10px;margin:3px;border:1px solid rgba(0,243,255,.3);'
                    f'border-radius:6px;color:#00f3ff;font-size:.72rem;text-decoration:none;'
                    f'font-family:monospace;background:rgba(0,243,255,.05)">{fname}</a>'
                )
        # Task result files
        results_dir = gd / "results"
        result_links = []
        if results_dir.exists():
            for rf in sorted(results_dir.glob("*_result.json")):
                url = f"/generations/{gen_num}/results/{rf.name}"
                task_id = rf.stem.replace("_result", "")
                # Try to read pass/fail
                try:
                    data = json.loads(rf.read_text())
                    icon = "✅" if data.get("success") else "❌"
                except Exception:
                    icon = "·"
                result_links.append(
                    f'<a href="{url}" target="_blank" style="display:inline-block;'
                    f'padding:4px 10px;margin:3px;border:1px solid rgba(255,255,255,.1);'
                    f'border-radius:6px;color:#e0e0e0;font-size:.68rem;text-decoration:none;'
                    f'font-family:monospace;background:rgba(255,255,255,.03)">'
                    f'{icon} {task_id}</a>'
                )
            # Summary files
            for sf in ["benchmark_summary.json", "manifest.json"]:
                if (results_dir / sf).exists():
                    url = f"/generations/{gen_num}/results/{sf}"
                    result_links.append(
                        f'<a href="{url}" target="_blank" style="display:inline-block;'
                        f'padding:4px 10px;margin:3px;border:1px solid rgba(255,179,0,.3);'
                        f'border-radius:6px;color:#ffb300;font-size:.68rem;text-decoration:none;'
                        f'font-family:monospace;background:rgba(255,179,0,.05)">{sf}</a>'
                    )

        sections.append(f"""
        <div style="background:rgba(15,16,35,.75);border:1px solid rgba(255,255,255,.08);
          border-radius:18px;padding:24px;margin-bottom:24px">
          <div style="font-size:1.4rem;font-weight:800;margin-bottom:6px;
            background:linear-gradient(90deg,#00f3ff,#ff00ff);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            {gen_num.replace("_"," ").upper()}
          </div>
          <div style="font-size:.58rem;color:#64748b;text-transform:uppercase;
            letter-spacing:2px;margin-bottom:16px">Agent Source Files</div>
          <div style="margin-bottom:20px">{''.join(agent_links) or '<span style="color:#64748b;font-size:.7rem">No files found</span>'}</div>
          <div style="font-size:.58rem;color:#64748b;text-transform:uppercase;
            letter-spacing:2px;margin-bottom:10px">Task Solutions (Generated Code)</div>
          <div>{''.join(result_links) or '<span style="color:#64748b;font-size:.7rem">No results yet</span>'}</div>
        </div>
        """)

    body = "\n".join(sections) if sections else (
        '<p style="color:#64748b">No generations found yet — evolution has not started.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Code Browser — Self-Replicating Agent</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;800&display=swap" rel="stylesheet">
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#05060f;color:#e0e0e0;font-family:'Outfit',sans-serif;padding:32px;
      min-height:100vh}}
    body::before{{content:'';position:fixed;inset:0;
      background-image:radial-gradient(circle at 2px 2px,rgba(255,255,255,.03) 1px,transparent 0);
      background-size:36px 36px;pointer-events:none;z-index:0}}
    .wrap{{max-width:900px;margin:0 auto;position:relative;z-index:1}}
    h1{{font-size:1.6rem;font-weight:800;margin-bottom:6px;
      background:linear-gradient(90deg,#00f3ff,#ff00ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .sub{{font-size:.7rem;color:#64748b;margin-bottom:32px}}
    .back{{display:inline-flex;align-items:center;gap:6px;margin-bottom:28px;
      color:#00f3ff;text-decoration:none;font-size:.75rem;
      border:1px solid rgba(0,243,255,.3);padding:5px 12px;border-radius:8px;
      background:rgba(0,243,255,.05)}}
    .back:hover{{background:rgba(0,243,255,.12)}}
  </style>
</head>
<body>
<div class="wrap">
  <a href="/MISSION_CONTROL.html" class="back">← Mission Control</a>
  <h1>Code Browser</h1>
  <div class="sub">Agent source files and task solutions for every generation.
    Click any file to view · JSON files open in browser · .py files download or open as text.</div>
  {body}
</div>
</body>
</html>"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files; handles /code browser and /api/* control endpoints."""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        clean = self.path.split("?")[0].rstrip("/")
        if clean == "/code":
            html = _build_code_browser().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif clean == "/api/models":
            body = json.dumps(GROQ_FREE_MODELS).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        global _restart_event, _current_proc
        clean = self.path.split("?")[0].rstrip("/")
        if clean != "/api/config":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) if length else b"{}")
        except Exception:
            self._json_response(400, {"error": "invalid JSON"})
            return

        action = body.get("action", "")

        if action == "set_model":
            model_id = body.get("model", "").strip()
            valid_ids = {m["id"] for m in GROQ_FREE_MODELS}
            if not model_id or model_id not in valid_ids:
                self._json_response(400, {"error": f"unknown model: {model_id}"})
                return
            data_dir = ROOT / "data"
            data_dir.mkdir(exist_ok=True)
            cfg_path = data_dir / "user_config.json"
            cfg_path.write_text(json.dumps({"model": model_id}))
            logger.info(f"[api] Model set to: {model_id}")
            self._json_response(200, {"ok": True, "model": model_id})

        elif action == "restart":
            logger.info("[api] Restart requested from dashboard")
            # Kill current subprocess, signal evolution loop to restart
            with _proc_lock:
                if _current_proc and _current_proc.poll() is None:
                    try:
                        _current_proc.terminate()
                        logger.info("[api] Terminated current evolution process")
                    except Exception as e:
                        logger.warning(f"[api] Could not terminate process: {e}")
            _restart_event.set()
            self._json_response(200, {"ok": True, "message": "Evolution restarting..."})

        else:
            self._json_response(400, {"error": f"unknown action: {action}"})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, fmt, *args):
        if len(args) > 1 and args[1] not in ("200", "204", "304"):
            super().log_message(fmt, *args)


def run_http_server():
    os.chdir(ROOT)
    server = http.server.HTTPServer(("0.0.0.0", PORT), QuietHandler)
    logger.info(f"Dashboard → http://0.0.0.0:{PORT}/MISSION_CONTROL.html")
    server.serve_forever()


def _clear_volatile_data():
    """Delete all runtime-generated data files so each run starts fresh."""
    results_dir = ROOT / "generations" / "gen_1" / "results"
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            f.unlink()

    VOLATILE_DATA = [
        "heartbeat.json", "evolution_log.json",
        "token_usage.json", "benchmark_matrix.json",
        "model_status.json",
    ]
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    for fname in VOLATILE_DATA:
        fp = data_dir / fname
        if fp.exists():
            fp.unlink()
            logger.info(f"  Cleared: data/{fname}")
    for fp in data_dir.glob("gen_*_progress.json"):
        fp.unlink()
        logger.info(f"  Cleared: data/{fp.name}")


def run_evolution():
    """
    Start Gen 1, then let each generation self-launch the next.
    Restarts from Gen 1 when:
      - The lineage ends (all generations complete/fail), after a 300s pause
      - The dashboard sends a restart request (_restart_event is set)
    """
    global _current_proc, _restart_event

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set — evolution cannot start. "
                     "Add it as an environment variable / secret.")
        return

    env = os.environ.copy()
    gen1_main = ROOT / "generations" / "gen_1" / "main.py"

    while True:
        _restart_event.clear()   # reset before each cycle

        if not gen1_main.exists():
            logger.error(f"Gen 1 main.py not found at {gen1_main}")
            time.sleep(60)
            continue

        logger.info("=== Starting evolution from Generation 1 ===")
        _clear_volatile_data()

        # Launch gen_1 as a Popen so we can kill it on dashboard restart
        proc = subprocess.Popen(
            [sys.executable, str(gen1_main), "--generation", "1"],
            cwd=ROOT / "generations" / "gen_1",
            env=env,
        )
        with _proc_lock:
            _current_proc = proc

        # Poll until process ends OR restart is requested
        while proc.poll() is None:
            if _restart_event.is_set():
                logger.info("[evolution] Dashboard restart requested — killing current run")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(2)

        exit_code = proc.returncode
        with _proc_lock:
            _current_proc = None

        if _restart_event.is_set():
            logger.info("=== Restarting evolution immediately (dashboard request) ===")
            _restart_event.clear()
            # Small delay so the dashboard sees the state change
            time.sleep(3)
        else:
            logger.info(f"Gen 1 process exited with code {exit_code}")
            logger.info("Waiting 300s before restarting evolution lineage...")
            # Wait but remain interruptible by restart event
            _restart_event.wait(timeout=300)
            if _restart_event.is_set():
                logger.info("=== Restarting early (dashboard request during wait) ===")
                _restart_event.clear()


if __name__ == "__main__":
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=False)
    http_thread.start()
    logger.info(f"HTTP server started on port {PORT}")

    # Small delay so server is ready before evolution logs start
    time.sleep(2)

    # Run evolution in foreground (keeps the process alive)
    run_evolution()
