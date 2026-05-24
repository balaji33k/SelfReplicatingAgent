import os
import shutil
import json

class Spawner:
    def __init__(self, current_dir, target_dir):
        self.current_dir = current_dir
        self.target_dir = target_dir
        
    def spawn(self, manifest, logic_files):
        os.makedirs(self.target_dir, exist_ok=True)
        REQUIRED = ["run.py", "meta_architect.py", "env.py", "spawner.py"]
        for filename, content in logic_files.items():
            if filename in REQUIRED:
                with open(os.path.join(self.target_dir, filename), "w") as f: f.write(content)
        for req in REQUIRED:
            if not os.path.exists(os.path.join(self.target_dir, req)):
                shutil.copy(os.path.join(self.current_dir, req), os.path.join(self.target_dir, req))
        shutil.copy(os.path.join(self.current_dir, "telemetry.py"), os.path.join(self.target_dir, "telemetry.py"))
        shutil.copy(os.path.join(self.current_dir, "problem_pool.json"), os.path.join(self.target_dir, "problem_pool.json"))
        with open(os.path.join(self.target_dir, "manifest.json"), "w") as f: json.dump(manifest, f, indent=4)
