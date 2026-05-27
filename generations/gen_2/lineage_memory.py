import json
import os
import logging

logger = logging.getLogger("LineageMemory")

MAX_REFERENCE_SOLUTIONS = 10


class LineageMemory:
    """
    Provides the lineage fallback feature: when a generation's pass rate
    falls below a configured threshold, the synthesis prompt is augmented
    with the reference generation's passing solutions and source architecture.

    Resolves the project root from __file__ so it works correctly whether
    this module sits at the project root or is copied into a generation dir.
    """

    def __init__(self):
        self_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(self_dir)
        if os.path.basename(parent_dir) == "generations":
            self.project_root = os.path.dirname(parent_dir)
        else:
            self.project_root = self_dir

    def is_enabled(self) -> bool:
        config = self._load_config()
        return config.get("features", {}).get("lineage_fallback", False)

    def should_consult(self, pass_percentage: float) -> bool:
        config = self._load_config()
        threshold = config.get("features", {}).get("fallback_threshold_pct", 50)
        return pass_percentage < threshold

    def get_reference_book(self) -> dict:
        config = self._load_config()
        ref_gen = config.get("features", {}).get("fallback_reference_gen", 1)

        performance = self._load_performance(ref_gen)
        architecture = self._load_source(ref_gen)
        passing_solutions = self._load_passing_solutions(ref_gen)

        if not architecture and not passing_solutions:
            logger.warning(f"Reference book for gen_{ref_gen} is empty — no useful data found")

        return {
            "reference_generation": ref_gen,
            "note": (
                "This is the reference generation's architecture and passing solutions. "
                "Study what worked. Understand why it worked. "
                "Do NOT copy — use this as a foundation to design something better."
            ),
            "performance": performance,
            "architecture": architecture,
            "passing_solutions": passing_solutions,
        }

    def _load_config(self) -> dict:
        path = os.path.join(self.project_root, "config.json")
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"config.json not found at {path} — lineage fallback disabled")
            return {}
        except Exception as e:
            logger.error(f"Failed to read config.json: {e}")
            return {}

    def _load_performance(self, gen_num: int) -> dict:
        path = os.path.join(self.project_root, "data", f"gen_{gen_num}.json")
        try:
            with open(path) as f:
                data = json.load(f)
            tasks = data.get("tasks", {})
            total = len(tasks)
            passed = sum(1 for t in tasks.values() if t.get("status") == "success")
            return {
                "pass_rate": f"{passed}/{total}",
                "pass_percentage": round(100 * passed / max(1, total), 1),
                "improvement_log": data.get("improvement_log", ""),
            }
        except FileNotFoundError:
            logger.warning(f"No telemetry data found for gen_{gen_num}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load gen_{gen_num} performance data: {e}")
            return {}

    def _load_source(self, gen_num: int) -> dict:
        gen_dir = os.path.join(self.project_root, "generations", f"gen_{gen_num}")
        source = {}
        if not os.path.exists(gen_dir):
            logger.warning(f"Gen {gen_num} directory not found at {gen_dir}")
            return source
        for fname in sorted(os.listdir(gen_dir)):
            if fname.endswith(".py") and fname not in ("telemetry.py", "lineage_memory.py"):
                try:
                    with open(os.path.join(gen_dir, fname)) as f:
                        source[fname] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read {fname} from gen_{gen_num}: {e}")
        return source

    def _load_passing_solutions(self, gen_num: int) -> dict:
        data_path = os.path.join(self.project_root, "data", f"gen_{gen_num}.json")
        sandbox_dir = os.path.join(
            self.project_root, "generations", f"gen_{gen_num}", "sandbox"
        )
        solutions = {}

        try:
            with open(data_path) as f:
                data = json.load(f)
            passing_ids = [
                task_id
                for task_id, t in data.get("tasks", {}).items()
                if t.get("status") == "success"
            ]
        except FileNotFoundError:
            logger.warning(f"No task data found for gen_{gen_num} — cannot load passing solutions")
            return solutions
        except Exception as e:
            logger.error(f"Failed to load gen_{gen_num} task data: {e}")
            return solutions

        for task_id in passing_ids[:MAX_REFERENCE_SOLUTIONS]:
            sol_path = os.path.join(sandbox_dir, f"sol_{task_id}.py")
            if os.path.exists(sol_path):
                try:
                    with open(sol_path) as f:
                        solutions[task_id] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read solution for {task_id}: {e}")

        logger.info(
            f"Loaded {len(solutions)}/{len(passing_ids)} passing solutions from gen_{gen_num}"
        )
        return solutions
