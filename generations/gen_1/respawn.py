"""
respawn.py — Manual respawn using existing benchmark results.

Use this when a generation completed its benchmark but SPAWN_FAILED.
Reads the existing results, re-runs evolution + spawn only.
Does NOT re-run any tasks.

Usage:
    cd generations/gen_1
    python respawn.py
"""
import json
import logging
import sys
from pathlib import Path

# Ensure gen_1 is on the path
GEN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GEN_DIR))

from config import Config
from analysis import FailureAnalyzer
from evolution_engine import EvolutionEngine
from spawner import Spawner
from utils import setup_logging, get_current_generation_dir

setup_logging("INFO", GEN_DIR / "respawn.log")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("RESPAWN — skipping benchmark, using existing results")
    logger.info("=" * 60)

    gen_config = Config()
    generation_number = 1
    current_gen_dir = get_current_generation_dir(generation_number)
    results_dir = current_gen_dir / "results"

    # ── Load existing task results ────────────────────────────────────
    result_files = sorted(results_dir.glob("*_result.json"))
    if not result_files:
        logger.error(f"No result files found in {results_dir}. Run the benchmark first.")
        sys.exit(1)

    all_results = []
    for rf in result_files:
        try:
            all_results.append(json.loads(rf.read_text()))
        except Exception as e:
            logger.warning(f"Could not read {rf.name}: {e}")

    logger.info(f"Loaded {len(all_results)} existing task results")

    # ── Re-run failure analysis ───────────────────────────────────────
    analyzer = FailureAnalyzer()
    analysis_report = analyzer.analyze(all_results)
    logger.info(
        f"Analysis: {analysis_report.passed}/{analysis_report.total_tasks} passed "
        f"({analysis_report.pass_rate:.1%})"
    )
    logger.info(f"Error breakdown: {analysis_report.failure_breakdown}")

    # Load parent pass rate from manifest if available
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        gen_config.parent_pass_rate = m.get("parent_pass_rate")
        gen_config.parent_error_types = m.get("error_types", {})

    # ── Re-run evolution engine ───────────────────────────────────────
    logger.info("Running evolution engine...")

    def _plan_cb(event, payload):
        if event == "plan_ready":
            logger.info(f"[evolution] Plan ready — {payload['files_total']} files to generate")
            logger.info(f"[evolution] Plan: {payload['plan'][:200]}")
        elif event == "file_done":
            logger.info(f"[evolution] [{payload['index']}/{payload['total']}] {payload['file']}")

    evolution_engine = EvolutionEngine(gen_config.llm_config)
    next_gen_design = evolution_engine.design_next_generation(
        analysis_report, gen_config, plan_callback=_plan_cb
    )

    next_gen_design.new_config.parent_pass_rate = analysis_report.pass_rate
    next_gen_design.new_config.parent_error_types = analysis_report.failure_breakdown
    if hasattr(gen_config, "topology"):
        next_gen_design.new_config._parent_topology = gen_config.topology.to_dict()
    next_gen_design.new_config._evolution_artifacts = next_gen_design.evolution_artifacts

    logger.info(f"Evolution complete — {len(next_gen_design.new_prompts_content)} files designed")
    logger.info(f"Improvement log: {next_gen_design.improvement_log_message[:200]}")

    # ── Spawn ─────────────────────────────────────────────────────────
    logger.info("Spawning Gen 2...")
    spawner = Spawner()
    next_gen_dir = spawner.spawn_next_generation(
        generation_number + 1,
        current_gen_dir,
        next_gen_design.new_config,
        next_gen_design.new_prompts_content,
        next_gen_design.improvement_log_message,
        analysis_report=analysis_report,
        task_results=all_results,
    )

    logger.info(f"✅ Gen 2 spawned at {next_gen_dir}")
    print(f"\n✅ Gen 2 spawned at {next_gen_dir}")
    print(f"   Files: {len(list(next_gen_dir.glob('*.py')))} .py files")
    print(f"   Run:   cd {next_gen_dir} && python main.py --generation 2")


if __name__ == "__main__":
    main()
