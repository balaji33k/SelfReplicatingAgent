import json
import os
import time
import subprocess
import telemetry
from env import Sandbox
from meta_architect import MetaArchitect
from spawner import Spawner

PYTHON_PATH = r"C:\Users\pc\.local\bin\python3.14.exe"

def main():
    gen_num = 3
    telemetry.update_industrial_dashboard(current_gen=gen_num, status="RUNNING")
    
    with open("problem_pool.json", "r") as f: pool = json.load(f)
    tasks = pool["training"]["swe"][:15] + pool["training"]["lcb"][:15]
    sandbox = Sandbox()
    ma = MetaArchitect()
    results = []
    
    for i, task in enumerate(tasks):
        code = ma.solve_task(task)
        res = sandbox.run(code, task["id"])
        results.append({"id": task["id"], "status": res["status"], "logs": res["logs"], "code": res["code"]})
        if (i+1) % 5 == 0:
            telemetry.update_industrial_dashboard(current_gen=gen_num, status="RUNNING", results=results)
            
    manifest, logic_files = ma.synthesize_offspring(gen_num, results)
    offspring_dir = os.path.join("..", f"gen_{gen_num + 1}")
    spawner = Spawner(".", offspring_dir)
    spawner.spawn(manifest, logic_files)
    
    telemetry.update_industrial_dashboard(current_gen=gen_num, status="COMPLETED", 
                                        results=results, 
                                        topology=manifest.get("TOPOLOGY"), 
                                        improvement_log=manifest.get("IMPROVEMENT_LOG"))
                                
    subprocess.Popen([PYTHON_PATH, "run.py"], cwd=offspring_dir)

if __name__ == "__main__":
    main()
