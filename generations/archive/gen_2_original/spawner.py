import os
import shutil
import json
import logging

logger = logging.getLogger("Spawner")

class Spawner:
    def __init__(self, current_dir, target_dir):
        self.current_dir = current_dir
        self.target_dir = target_dir

    def spawn(self, manifest, logic_files):
        REQUIRED = ["run.py", "meta_architect.py", "env.py", "spawner.py"]

        missing = [r for r in REQUIRED if r not in logic_files]
        if missing:
            raise RuntimeError(
                f"LLM did not generate required files: {missing}. "
                "Offspring spawn aborted. Check synthesize_offspring() output."
            )

        os.makedirs(self.target_dir, exist_ok=True)

        for filename in REQUIRED:
            path = os.path.join(self.target_dir, filename)
            with open(path, "w") as f:
                f.write(logic_files[filename])
            logger.info(f"Written LLM-generated {filename} → {path}")

        # telemetry.py is dashboard infrastructure — copy, don't ask LLM to regenerate it
        telemetry_src = os.path.join(self.current_dir, "telemetry.py")
        if os.path.exists(telemetry_src):
            shutil.copy(telemetry_src, os.path.join(self.target_dir, "telemetry.py"))

        # Prefer the enriched problem pool from data/ over the parent's local copy
        project_root = os.path.abspath(os.path.join(self.current_dir, "..", ".."))
        enriched_pool = os.path.join(project_root, "data", "fixed_problem_pool.json")
        if os.path.exists(enriched_pool):
            shutil.copy(enriched_pool, os.path.join(self.target_dir, "problem_pool.json"))
            logger.info(f"Copied enriched problem pool from {enriched_pool}")
        else:
            parent_pool = os.path.join(self.current_dir, "problem_pool.json")
            shutil.copy(parent_pool, os.path.join(self.target_dir, "problem_pool.json"))
            logger.warning("Enriched pool not found — falling back to parent's problem_pool.json")

        with open(os.path.join(self.target_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        logger.info(
            f"Generation {manifest['GENERATION']} spawned at {self.target_dir} "
            f"(parent pass rate: {manifest.get('PARENT_PASS_RATE', 'N/A')})"
        )
