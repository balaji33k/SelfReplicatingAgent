"""
main.py — Entry point for a single generation run.

Replaces the monolithic CodeGenerator+SandboxExecutor loop with the
7-agent specialization pipeline (AgentPipeline).

Flow per task:
  TaskContract → Analyst → Architect → Coder → Critic → Reviser → Executor → Debugger
  → PipelineResult → FailureAnalyzer → EvolutionEngine → Spawner → next gen
"""
import argparse
import logging
import os
import signal
import sys
import json
from pathlib import Path

from config import Config
from utils import setup_logging, get_current_generation_dir, get_next_generation_dir, save_json
from task_manager import TaskLoader
from analysis import FailureAnalyzer
from evolution_engine import EvolutionEngine
from spawner import Spawner
from llm_client import LLMClient
from pipeline import AgentPipeline
from contracts import TaskContract
from evolution_tracker import record_generation_complete

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (generations/gen_N/ → project/)
_GEN_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _GEN_DIR.parent.parent if _GEN_DIR.parent.name == "generations" else _GEN_DIR.parent
_HEARTBEAT_PATH = _PROJECT_ROOT / "data" / "heartbeat.json"
_STOP_FLAG      = _PROJECT_ROOT / "data" / "stop.flag"
_PID_FILE       = _PROJECT_ROOT / "data" / "active_pid.json"
_GEN_TOPOLOGY: dict = {}  # set once in run_generation, included in every heartbeat write


def _register_pid(gen_num: int) -> None:
    """Write current PID to data/active_pid.json so the dashboard can kill this process."""
    try:
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(json.dumps({"pid": os.getpid(), "gen": gen_num}))
    except Exception:
        pass


def _check_stop_flag(gen_num: int, done: int, total: int, passed: int, failed: int) -> None:
    """If data/stop.flag exists, write STOPPED heartbeat and exit cleanly."""
    if _STOP_FLAG.exists():
        logger.info("[main] Stop flag detected — halting evolution gracefully.")
        _write_heartbeat(gen_num, "STOPPED", done, total, passed, failed, "", [])
        sys.exit(0)


def _probe_apis(llm_client, out_path):
    """Fire a tiny test prompt at each provider and record latency/error to out_path."""
    import concurrent.futures as _cf, time as _t
    results = {}
    test_prompt = "Reply with one word: hello"
    for model_id in ["gemini-2.0-flash", "meta-llama/llama-4-scout-17b-16e-instruct"]:
        t0 = _t.time()
        try:
            old_model = llm_client.config.model_name
            llm_client._switch_to_model(model_id)
            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            from langchain_core.messages import HumanMessage
            _fut = _ex.submit(llm_client._llm.invoke, [HumanMessage(content=test_prompt)])
            try:
                resp = _fut.result(timeout=20)
                results[model_id] = {"ok": True, "latency": round(_t.time()-t0,2), "snippet": str(getattr(resp,"content",""))[:40]}
            except _cf.TimeoutError:
                results[model_id] = {"ok": False, "error": "timeout_20s", "latency": 20}
            finally:
                _ex.shutdown(wait=False)
            llm_client._switch_to_model(old_model)
        except Exception as e:
            results[model_id] = {"ok": False, "error": str(e)[:120], "latency": round(_t.time()-t0,2)}
    try:
        import json as _j
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_j.dumps({"probed_at": __import__("datetime").datetime.utcnow().isoformat()+"Z", "results": results}, indent=2))
        logger.info(f"[probe] API connectivity: {results}")
    except Exception:
        pass


def _write_progress(gen_num: int, task_ids: list, completed: dict, descriptions: dict = None) -> None:
    """Write per-task progress to data/gen_N_progress.json — read by the dashboard."""
    try:
        _HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        path = _HEARTBEAT_PATH.parent / f"gen_{gen_num}_progress.json"
        path.write_text(json.dumps({
            "gen": gen_num,
            "task_list": task_ids,
            "results": completed,
            "descriptions": descriptions or {},
        }))
    except Exception:
        pass


def _write_heartbeat(
    gen_num: int,
    status: str,
    done: int,
    total: int,
    passed: int,
    failed: int,
    active_task_id: str = "",
    active_agents: list = None,
    current_phase: str = "",
    phase_results: dict = None,
) -> None:
    """Write live status to data/heartbeat.json — read by the dashboard.

    current_phase: one of thinking | writing_unit_tests | writing_code |
                   reviewing | revising | compiling | running_unit_tests |
                   debugging | running_integration_tests
    phase_results: {"compile": bool|None, "unit_tests": bool|None,
                    "integration_tests": bool|None}
    """
    try:
        import datetime as _dt
        _HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_PATH.write_text(json.dumps({
            "active_gen": gen_num,
            "status": status,
            "progress": f"{done}/{total}",
            "progress_pct": round(done / total, 3) if total else 0,
            "active_task_id": active_task_id,
            "passed_so_far": passed,
            "failed_so_far": failed,
            "pass_rate_so_far": round(passed / done, 3) if done else 0,
            "active_agents": active_agents or [],
            "current_phase": current_phase,
            "phase_results": phase_results or {},
            "topology": _GEN_TOPOLOGY,
            "last_updated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }))
    except Exception:
        pass  # heartbeat is best-effort — never block the main loop


def _task_to_contract(task) -> TaskContract:
    """Convert a task_manager.Task to a contracts.TaskContract."""
    # task_manager.Task has: task_id, description, test_cases, signature, metadata
    return TaskContract(
        task_id=task.task_id,
        problem_statement=getattr(task, "description", "") or getattr(task, "problem_statement", ""),
        test_cases=task.test_cases or [],
        signature=getattr(task, "signature", None),
        metadata=getattr(task, "metadata", {}),
    )


def run_generation(gen_config: Config, generation_number: int):
    """
    Orchestrates the entire evolutionary loop for a single generation.
    Uses the multi-agent specialization pipeline for each task.
    """
    global _GEN_TOPOLOGY
    _GEN_TOPOLOGY = gen_config.topology.to_dict() if hasattr(gen_config.topology, "to_dict") else {}

    current_gen_dir = get_current_generation_dir(generation_number)
    results_dir = current_gen_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve pool: env var override, then gen dir pool, then project root pool
    pool_path = Path(os.getenv("PROBLEM_POOL_PATH", current_gen_dir / "problem_pool.json"))

    logger.info(f"--- Running Generation {generation_number} ---")
    logger.info(f"Loading tasks from: {pool_path}")

    # 1. Task Ingestion
    task_loader = TaskLoader(pool_path)
    tasks = task_loader.load_tasks()
    if not tasks:
        logger.error("No tasks loaded. Aborting generation run.")
        sys.exit(1)
    logger.info(f"Loaded {len(tasks)} tasks.")

    # 2. Build multi-agent pipeline
    llm_client = LLMClient(gen_config.llm_config)

    # Quick API connectivity probe — writes data/api_probe.json so we can diagnose hangs
    _probe_apis(llm_client, _PROJECT_ROOT / "data" / "api_probe.json")

    pipeline = AgentPipeline(llm_client, gen_config.topology)
    logger.info(
        f"Pipeline topology: critic={gen_config.topology.enable_critic}, "
        f"reviser={gen_config.topology.enable_reviser}, "
        f"test_writer={gen_config.topology.enable_test_writer}, "
        f"debugger={gen_config.topology.enable_debugger}"
    )

    all_execution_results = []
    passed_count = 0
    failed_count = 0
    total_tasks = len(tasks)
    task_ids = [t.task_id for t in tasks]
    completed_progress: dict = {}

    # Build short descriptions for each task (first line, max 80 chars)
    task_descriptions: dict = {}
    for t in tasks:
        raw = getattr(t, "description", "") or getattr(t, "problem_statement", "") or ""
        first_line = raw.strip().split("\n")[0].strip()
        task_descriptions[t.task_id] = first_line[:80] if first_line else t.task_id

    # Register PID so the dashboard Stop button can kill this process directly
    _register_pid(generation_number)

    _write_heartbeat(generation_number, "RUNNING", 0, total_tasks, 0, 0, "", [])
    _write_progress(generation_number, task_ids, {}, task_descriptions)

    # 3. Run pipeline for each task
    for i, task in enumerate(tasks):
        # Check stop flag before each task — allows clean mid-benchmark stop
        _check_stop_flag(generation_number, i, total_tasks, passed_count, failed_count)

        logger.info(f"[{i+1}/{total_tasks}] Processing task: {task.task_id}")
        _write_heartbeat(
            generation_number, "RUNNING", i, total_tasks,
            passed_count, failed_count, task.task_id, [],
            current_phase="thinking", phase_results={}
        )

        # Phase callback — fires before each pipeline stage; updates heartbeat live.
        # Use default-arg capture for i/task_id (loop values) so each closure is
        # independent. passed_count / failed_count are read by reference so they
        # reflect the latest counts when the callback fires mid-task.
        def _make_phase_cb(_i=i, _tid=task.task_id):
            def _phase_cb(phase: str, ph_results: dict = None):
                _write_heartbeat(
                    generation_number, "RUNNING", _i, total_tasks,
                    passed_count, failed_count, _tid, [],
                    current_phase=phase, phase_results=ph_results or {}
                )
            return _phase_cb

        phase_cb = _make_phase_cb()

        # ── Per-task hard timeout (SIGALRM) ────────────────────────────
        # SIGALRM fires at the OS level and interrupts ANY blocking call
        # including C-extension network I/O — unlike thread-based timeouts.
        # 240s = 4 min max per task (covers multiple model fallback retries).
        _TASK_TIMEOUT_SEC = 360   # 6 × 60s = covers all 5 models + recovery

        def _task_alarm_handler(signum, frame):
            raise TimeoutError(
                f"Task {task.task_id} exceeded {_TASK_TIMEOUT_SEC}s hard limit"
            )

        signal.signal(signal.SIGALRM, _task_alarm_handler)
        signal.alarm(_TASK_TIMEOUT_SEC)

        try:
            contract = _task_to_contract(task)
            result = pipeline.solve(contract, phase_callback=phase_cb)
            signal.alarm(0)  # cancel alarm on success

            result_dict = result.to_dict()
            all_execution_results.append(result_dict)
            save_json(result_dict, results_dir / f"{task.task_id}_result.json")

            if result.success:
                passed_count += 1
            else:
                failed_count += 1

            # Update per-task progress file so dashboard can show live task list
            completed_progress[task.task_id] = {
                "status": "pass" if result.success else "fail",
                "error_type": result.error_type or "",
                "runtime": round(result.runtime, 2),
            }
            _write_progress(generation_number, task_ids, completed_progress, task_descriptions)

            status = "PASS" if result.success else f"FAIL ({result.error_type})"
            cycle_info = ", ".join(f"{k}={v}" for k, v in result.cycles.items() if v > 0)
            logger.info(
                f"Task {task.task_id}: {status}"
                + (f" | cycles: {cycle_info}" if cycle_info else "")
                + (f" | agents: {result.agents_used}" if not result.success else "")
            )

        except TimeoutError as e:
            signal.alarm(0)  # cancel alarm
            logger.error(f"Task {task.task_id} TIMED OUT: {e} — marking as failed and continuing")
            failed_count += 1
            completed_progress[task.task_id] = {
                "status": "fail",
                "error_type": "timeout",
                "runtime": _TASK_TIMEOUT_SEC,
            }
            _write_progress(generation_number, task_ids, completed_progress, task_descriptions)
        except Exception as e:
            signal.alarm(0)  # cancel alarm
            logger.exception(f"Unexpected error for task {task.task_id}: {e}")
            failed_count += 1
            error_result = {
                "task_id": task.task_id,
                "success": False,
                "final_code": "",
                "stdout": "",
                "stderr": f"Internal pipeline error: {str(e)}",
                "error_type": "SystemError",
                "runtime": 0.0,
                "agents_used": [],
                "cycles": {},
                "agent_failures": {"pipeline": str(e)},
            }
            all_execution_results.append(error_result)
            save_json(error_result, results_dir / f"{task.task_id}_result.json")

    # Check stop flag one final time before entering ANALYSING / spawning
    _check_stop_flag(generation_number, total_tasks, total_tasks, passed_count, failed_count)

    _write_heartbeat(generation_number, "ANALYSING", total_tasks, total_tasks,
                     passed_count, failed_count, "", [])

    if not all_execution_results:
        logger.error("No execution results recorded. Cannot proceed with analysis.")
        sys.exit(1)

    # 4. Failure Analysis
    failure_analyzer = FailureAnalyzer()
    analysis_report = failure_analyzer.analyze_results(all_execution_results, tasks)
    save_json(analysis_report.to_dict(), results_dir / "benchmark_summary.json")
    logger.info(
        f"Benchmark Summary: Passed {analysis_report.passed}/{analysis_report.total_tasks} "
        f"({analysis_report.pass_rate:.2%}). Failures: {analysis_report.failure_breakdown}"
    )

    # Record current generation's manifest
    current_manifest = {
        "generation": generation_number,
        "parent_pass_rate": getattr(gen_config, "parent_pass_rate", None),
        "parent_error_types": getattr(gen_config, "parent_error_types", {}),
        "pass_rate": analysis_report.pass_rate,
        "error_types": analysis_report.failure_breakdown,
        "timestamp": gen_config.timestamp,
        "improvement_log": gen_config.improvement_log,
        "topology": gen_config.topology.to_dict(),
    }
    save_json(current_manifest, results_dir / "manifest.json")

    # Guardrail: No Unchecked Spawning
    if analysis_report.total_tasks == 0 or analysis_report.pass_rate is None:
        logger.error("Analysis report is incomplete. Cannot spawn next generation.")
        sys.exit(1)

    # Guardrail: No Regression Without Record
    if (
        current_manifest["parent_pass_rate"] is not None
        and current_manifest["pass_rate"] < current_manifest["parent_pass_rate"]
    ):
        logger.warning(
            f"Regression detected: Current pass rate {current_manifest['pass_rate']:.2%} "
            f"is lower than parent's {current_manifest['parent_pass_rate']:.2%}."
        )

    # 5. Evolutionary Design
    evolution_engine = EvolutionEngine(gen_config.llm_config)
    try:
        next_gen_design = evolution_engine.design_next_generation(analysis_report, gen_config)

        # Propagate parent performance metrics into next gen config
        next_gen_design.new_config.parent_pass_rate = analysis_report.pass_rate
        next_gen_design.new_config.parent_error_types = analysis_report.failure_breakdown
        # Stash parent topology so evolution_tracker can diff it
        if hasattr(gen_config, "topology"):
            next_gen_design.new_config._parent_topology = gen_config.topology.to_dict()

    except Exception as e:
        logger.exception(f"Evolutionary design failed: {e}")
        sys.exit(1)

    # 5b. Back-fill this generation's actual pass rate into the previous evolution entry
    try:
        record_generation_complete(generation_number, analysis_report.pass_rate)
    except Exception as e:
        logger.warning(f"record_generation_complete failed (non-fatal): {e}")

    # 6. Offspring Spawning
    spawner = Spawner()
    try:
        _write_heartbeat(generation_number, "SPAWNING", total_tasks, total_tasks,
                         passed_count, failed_count, "", [])
        next_gen_dir = spawner.spawn_next_generation(
            generation_number + 1,
            current_gen_dir,
            next_gen_design.new_config,
            next_gen_design.new_prompts_content,
            next_gen_design.improvement_log_message,
            analysis_report=analysis_report,
            task_results=all_execution_results,
        )
        logger.info(
            f"Successfully spawned Generation {generation_number + 1} at {next_gen_dir}"
        )
        logger.info(
            f"Next generation improvement log: {next_gen_design.improvement_log_message}"
        )

        # Launch the next generation as an independent process
        spawner.launch_next_generation(next_gen_dir)
        _write_heartbeat(generation_number, "DONE", total_tasks, total_tasks,
                         passed_count, failed_count, "", [])

    except Exception as e:
        logger.exception(f"Offspring spawning failed: {e}")
        _write_heartbeat(generation_number, "SPAWN_FAILED", total_tasks, total_tasks,
                         passed_count, failed_count, "", [])
        sys.exit(1)

    finally:
        # Persist all generation artifacts to HuggingFace Dataset repo
        # so they survive Space restarts and are browsable from HF.
        # Non-fatal: evolution is already done at this point.
        try:
            from persist import upload_generation
            logger.info(f"[persist] Uploading Gen {generation_number} to HuggingFace Dataset…")
            upload_generation(generation_number, current_gen_dir)
        except Exception as e:
            logger.warning(f"[persist] Upload skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a generation of the self-replicating AI coding agent."
    )
    parser.add_argument(
        "--generation", type=int, default=1, help="The current generation number."
    )
    args = parser.parse_args()

    # Load configuration for the current generation
    current_gen_dir = get_current_generation_dir(args.generation)

    # Ensure this gen's directory is on the path so local imports resolve
    sys.path.insert(0, str(current_gen_dir))

    gen_config = Config()
    setup_logging(gen_config.log_level, current_gen_dir / "generation.log")

    try:
        run_generation(gen_config, args.generation)
    except Exception as main_exc:
        logger.exception(
            f"Unhandled exception in main for Generation {args.generation}: {main_exc}"
        )
        sys.exit(1)
