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


AGENT_FILES = ["main.py", "pipeline.py", "meta_architect.py", "spawner.py",
               "env.py", "config.py", "contracts.py", "llm_client.py",
               "evolution_tracker.py", "task_manager.py", "analysis.py",
               "prompts.py", "telemetry.py"]


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
    """Serve static files with no-cache headers; also handles /code browser page."""

    def do_GET(self):
        clean = self.path.split("?")[0].rstrip("/")
        if clean == "/code":
            html = _build_code_browser().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html)
        else:
            super().do_GET()

    def end_headers(self):
        # Prevent CDN/browser from caching live JSON data
        path = self.path.split("?")[0]
        if "/data/" in path or path.endswith(".json"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        if len(args) > 1 and args[1] not in ("200", "304"):
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
                     "Add it as an environment variable / secret.")
        return

    env = os.environ.copy()
    gen1_main = ROOT / "generations" / "gen_1" / "main.py"

    while True:
        if not gen1_main.exists():
            logger.error(f"Gen 1 main.py not found at {gen1_main}")
            time.sleep(60)
            continue

        logger.info("=== Starting evolution from Generation 1 ===")

        # Clean previous results so each run is fresh
        results_dir = ROOT / "generations" / "gen_1" / "results"
        if results_dir.exists():
            for f in results_dir.glob("*.json"):
                f.unlink()

        proc = subprocess.run(
            [sys.executable, str(gen1_main), "--generation", "1"],
            cwd=ROOT / "generations" / "gen_1",
            env=env,
        )

        exit_code = proc.returncode
        logger.info(f"Gen 1 process exited with code {exit_code}")
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
