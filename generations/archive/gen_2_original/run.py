import json
import os
import sys
import subprocess
import logging
import telemetry
from env import Sandbox
from meta_architect import MetaArchitect
from spawner import Spawner

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("Run")

GEN_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    gen_num = 2

    telemetry.update_industrial_dashboard(current_gen=gen_num, status="RUNNING")

    pool_path = os.path.join(GEN_DIR, "problem_pool.json")
    with open(pool_path, "r") as f:
        pool = json.load(f)

    tasks = pool["training"]["swe"][:15] + pool["training"]["lcb"][:15]
    logger.info(f"Gen {gen_num}: loaded {len(tasks)} tasks")

    sandbox = Sandbox()
    ma = MetaArchitect()
    results = []

    for i, task in enumerate(tasks):
        logger.info(f"Task {i+1}/{len(tasks)}: {task['id']}")
        code = ma.solve_task(task)
        res = sandbox.run(code, task["id"])
        results.append({
            "id": task["id"],
            "status": res["status"],
            "logs": res["logs"],
            "code": res["code"]
        })
        if (i + 1) % 5 == 0:
            telemetry.update_industrial_dashboard(current_gen=gen_num, status="RUNNING", results=results)

    passed = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Gen {gen_num} benchmark complete: {passed}/{len(results)} passed")

    manifest, logic_files = ma.synthesize_offspring(gen_num, results)

    offspring_dir = os.path.abspath(os.path.join(GEN_DIR, "..", f"gen_{gen_num + 1}"))
    spawner = Spawner(GEN_DIR, offspring_dir)
    spawner.spawn(manifest, logic_files)

    telemetry.update_industrial_dashboard(
        current_gen=gen_num,
        status="COMPLETED",
        results=results,
        topology=manifest.get("TOPOLOGY"),
        improvement_log=manifest.get("IMPROVEMENT_LOG")
    )

    logger.info(f"Spawning Gen {gen_num + 1} at {offspring_dir}")
    subprocess.Popen([sys.executable, "run.py"], cwd=offspring_dir)

if __name__ == "__main__":
    main()
