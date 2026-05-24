import subprocess
import sys
import os
import logging

logger = logging.getLogger("Sandbox")

class Sandbox:
    def __init__(self):
        self.sandbox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")

    def run(self, code, task_id):
        os.makedirs(self.sandbox_dir, exist_ok=True)
        sol_file = os.path.join(self.sandbox_dir, f"sol_{task_id}.py")

        clean_code = self._strip_markdown(code)

        with open(sol_file, "w") as f:
            f.write(clean_code)

        try:
            res = subprocess.run(
                [sys.executable, sol_file],
                capture_output=True,
                text=True,
                timeout=15
            )
            logs = f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            status = "success" if res.returncode == 0 else "failed"
            return {"status": status, "logs": logs, "code": clean_code}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "logs": "Timeout after 15s", "code": clean_code}
        except Exception as e:
            logger.error(f"Sandbox error for {task_id}: {e}")
            return {"status": "error", "logs": str(e), "code": clean_code}

    def _strip_markdown(self, code):
        if "```python" in code:
            return code.split("```python")[1].split("```")[0].strip()
        if "```" in code:
            return code.split("```")[1].split("```")[0].strip()
        return code.strip()
