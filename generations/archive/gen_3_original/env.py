import subprocess
import os

class Sandbox:
    def run(self, code, task_id):
        os.makedirs("sandbox", exist_ok=True)
        sol_file = f"sandbox/sol_{task_id}.py"
        with open(sol_file, "w") as f: f.write(code)
        try:
            res = subprocess.run([r"C:\Users\pc\.local\bin\python3.14.exe", sol_file], capture_output=True, text=True, timeout=5)
            logs = f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            return {"status": "success" if res.returncode == 0 else "failed", "logs": logs, "code": code}
        except subprocess.TimeoutExpired: return {"status": "timeout", "logs": "Timeout after 5s", "code": code}
