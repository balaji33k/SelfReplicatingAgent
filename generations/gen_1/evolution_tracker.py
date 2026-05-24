"""
evolution_tracker.py — Living evolution document.

Called by the Spawner at every spawn event. Maintains:
  1. EVOLUTION_LOG.md       — human-readable append-only chronicle at project root
  2. data/evolution_log.json — machine-readable entries for the live dashboard
  3. data/benchmark_matrix.json — cross-generation task × result matrix

Also called by main.py after each generation completes to back-fill the
"actual" pass rate into the previous evolution entry (predicted vs actual).
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Resolve project root: generations/gen_N/  →  project/"""
    here = Path(__file__).resolve().parent          # gen_N/
    gens = here.parent                               # generations/
    if gens.name == "generations":
        return gens.parent                           # project root
    return here.parent


def _data_dir() -> Path:
    d = _project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Public API ─────────────────────────────────────────────────────────────────

def record_spawn(
    from_gen: int,
    analysis_report,                  # AnalysisReport from analysis.py
    task_results: List[Dict],         # raw result dicts from main.py
    improvement_log: str,
    topology_from: Dict,
    topology_to: Dict,
    originality_report: Optional[Dict] = None,  # from CloneInspectorAgent
) -> None:
    """
    Called by Spawner immediately after validation passes.
    Records what failed in gen N and what is changing in gen N+1.
    """
    to_gen = from_gen + 1
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = _build_entry(
        from_gen=from_gen,
        to_gen=to_gen,
        timestamp=timestamp,
        analysis_report=analysis_report,
        task_results=task_results,
        improvement_log=improvement_log,
        topology_from=topology_from,
        topology_to=topology_to,
        originality_report=originality_report or {},
    )

    _append_evolution_log(entry)
    _update_benchmark_matrix(from_gen, task_results)
    _append_markdown(entry)

    logger.info(
        f"[evolution_tracker] Recorded Gen {from_gen} → {to_gen} "
        f"| pass_rate={entry['pass_rate_actual']:.1%} "
        f"| top_error={_top_error(analysis_report)}"
    )


def record_generation_complete(gen_num: int, pass_rate: float) -> None:
    """
    Called by main.py when a generation finishes its benchmark.
    Back-fills the 'pass_rate_predicted' field in the PREVIOUS entry
    (i.e. the entry that spawned this generation) with the actual result.
    """
    log_path = _data_dir() / "evolution_log.json"
    if not log_path.exists():
        return

    try:
        entries = json.loads(log_path.read_text())
        for entry in entries:
            if entry.get("to_gen") == gen_num and entry.get("pass_rate_predicted") is None:
                entry["pass_rate_predicted"] = pass_rate
                delta = pass_rate - entry.get("pass_rate_actual", 0)
                entry["improvement_delta"] = round(delta, 4)
                entry["improved"] = delta > 0
                break
        log_path.write_text(json.dumps(entries, indent=2))
        logger.info(
            f"[evolution_tracker] Back-filled Gen {gen_num} actual pass_rate={pass_rate:.1%}"
        )
    except Exception as e:
        logger.warning(f"[evolution_tracker] Could not back-fill Gen {gen_num}: {e}")


# ── Build entry ────────────────────────────────────────────────────────────────

def _build_entry(
    from_gen, to_gen, timestamp, analysis_report,
    task_results, improvement_log, topology_from, topology_to,
    originality_report: Dict = None,
) -> Dict[str, Any]:
    pass_rate = getattr(analysis_report, "pass_rate", 0.0)
    failure_breakdown = getattr(analysis_report, "failure_breakdown", {})
    agent_failure_counts = getattr(analysis_report, "agent_failure_counts", {})
    sample_failures = getattr(analysis_report, "sample_failures", [])

    # Per-task summary rows
    task_rows = []
    for r in task_results:
        task_rows.append({
            "task_id": r.get("task_id", "?"),
            "status": "pass" if r.get("success") else "fail",
            "error_type": r.get("error_type") or "",
            "agents_used": r.get("agents_used", []),
            "cycles": r.get("cycles", {}),
        })

    # Topology diff: which agents changed?
    topo_changes = _diff_topology(topology_from, topology_to)

    return {
        "from_gen": from_gen,
        "to_gen": to_gen,
        "timestamp": timestamp,
        "pass_rate_actual": round(pass_rate, 4),
        "pass_rate_predicted": None,       # filled when to_gen completes
        "improvement_delta": None,
        "improved": None,
        "passed": getattr(analysis_report, "passed", 0),
        "total": getattr(analysis_report, "total_tasks", 0),
        "failure_breakdown": failure_breakdown,
        "agent_failure_counts": agent_failure_counts,
        "top_failures": sample_failures[:5],
        "improvement_log": improvement_log,
        "topology_from": topology_from,
        "topology_to": topology_to,
        "topology_changes": topo_changes,
        "task_results": task_rows,
        "originality": originality_report or {},
    }


def _diff_topology(t_from: Dict, t_to: Dict) -> List[str]:
    changes = []
    agent_flags = ["enable_critic", "enable_reviser", "enable_test_writer", "enable_debugger"]
    labels = {
        "enable_critic": "Critic",
        "enable_reviser": "Reviser",
        "enable_test_writer": "TestWriter",
        "enable_debugger": "Debugger",
    }
    for flag in agent_flags:
        was = t_from.get(flag, False)
        now = t_to.get(flag, False)
        if was != now:
            verb = "enabled" if now else "disabled"
            changes.append(f"{labels[flag]} {verb}")
    for key in ("max_revise_cycles", "max_debug_cycles", "execution_timeout"):
        was = t_from.get(key)
        now = t_to.get(key)
        if was is not None and now is not None and was != now:
            changes.append(f"{key}: {was} → {now}")
    return changes


def _top_error(analysis_report) -> str:
    bd = getattr(analysis_report, "failure_breakdown", {})
    if not bd:
        return "none"
    return max(bd, key=bd.get)


# ── Persist evolution log ──────────────────────────────────────────────────────

def _append_evolution_log(entry: Dict) -> None:
    log_path = _data_dir() / "evolution_log.json"
    try:
        entries = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        entries = []
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2))


# ── Benchmark matrix ───────────────────────────────────────────────────────────

def _update_benchmark_matrix(gen_num: int, task_results: List[Dict]) -> None:
    """
    Upsert gen N's results into the cross-generation matrix.
    Format:
      { "tasks": ["t1", "t2", ...],
        "gens": ["1", "2", ...],
        "matrix": { "t1": {"1": "pass", "2": "fail"}, ... } }
    """
    matrix_path = _data_dir() / "benchmark_matrix.json"
    try:
        matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {}
    except Exception:
        matrix = {}

    if "matrix" not in matrix:
        matrix = {"tasks": [], "gens": [], "matrix": {}}

    gen_key = str(gen_num)
    if gen_key not in matrix["gens"]:
        matrix["gens"].append(gen_key)
        matrix["gens"].sort(key=int)

    for r in task_results:
        tid = r.get("task_id", "?")
        status = "pass" if r.get("success") else "fail"
        error = r.get("error_type") or ""

        if tid not in matrix["tasks"]:
            matrix["tasks"].append(tid)
        if tid not in matrix["matrix"]:
            matrix["matrix"][tid] = {}
        matrix["matrix"][tid][gen_key] = {"status": status, "error": error}

    # Annotate first-solved generation
    for tid, gen_data in matrix["matrix"].items():
        first_pass = next(
            (g for g in sorted(gen_data, key=int) if gen_data[g]["status"] == "pass"), None
        )
        matrix["matrix"][tid]["_first_pass_gen"] = first_pass

    matrix_path.write_text(json.dumps(matrix, indent=2))


# ── Markdown document ──────────────────────────────────────────────────────────

def _append_markdown(entry: Dict) -> None:
    md_path = _project_root() / "EVOLUTION_LOG.md"

    lines = []

    # Header (only on first entry)
    if not md_path.exists():
        lines += [
            "# Evolution Log\n",
            "Living document — appended at every generation spawn.\n",
            "Each entry records what failed, why, and what the next generation changes.\n\n",
            "---\n\n",
        ]

    from_gen = entry["from_gen"]
    to_gen = entry["to_gen"]
    ts = entry["timestamp"][:19].replace("T", " ")
    pct = f"{entry['pass_rate_actual']:.1%}"

    lines.append(f"## Generation {from_gen} → {to_gen}  |  {ts}\n\n")
    lines.append(f"**Pass rate:** {entry['passed']}/{entry['total']} ({pct})\n\n")

    # Benchmark results table
    lines.append("### Benchmark Results — Gen " + str(from_gen) + "\n\n")
    lines.append("| Task | Status | Error | Agents Used |\n")
    lines.append("|---|---|---|---|\n")
    for row in entry["task_results"]:
        status_icon = "✅" if row["status"] == "pass" else "❌"
        agents = " → ".join(row["agents_used"]) if row["agents_used"] else "—"
        error = row["error_type"] or "—"
        lines.append(f"| {row['task_id']} | {status_icon} | {error} | {agents} |\n")
    lines.append("\n")

    # Failure breakdown
    if entry["failure_breakdown"]:
        lines.append("### Failure Breakdown\n\n")
        lines.append("| Error Type | Count |\n")
        lines.append("|---|---|\n")
        for etype, count in sorted(entry["failure_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"| {etype} | {count} |\n")
        lines.append("\n")

    # Agent attribution
    if entry["agent_failure_counts"]:
        lines.append("### Agent Failure Attribution\n\n")
        for agent, count in sorted(entry["agent_failure_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{agent}**: contributed to {count} failure(s)\n")
        lines.append("\n")

    # Topology changes
    if entry["topology_changes"]:
        lines.append("### Topology Changes Gen " + str(from_gen) + " → " + str(to_gen) + "\n\n")
        for change in entry["topology_changes"]:
            lines.append(f"- {change}\n")
        lines.append("\n")
    else:
        lines.append(f"### Topology Changes\n\nNone — same pipeline structure.\n\n")

    # Clone Inspector report
    orig = entry.get("originality", {})
    if orig:
        verdict = orig.get("verdict", "?")
        score = orig.get("originality_score", None)
        score_str = f" ({score:.0%})" if isinstance(score, float) else ""
        lines.append(f"### Clone Inspector — Verdict: {verdict}{score_str}\n\n")
        if orig.get("llm_assessment"):
            lines.append(orig["llm_assessment"][:500] + "\n\n")
        if orig.get("near_clone_files"):
            lines.append(f"⚠️  Near-clone files: `{'`, `'.join(orig['near_clone_files'])}`\n\n")
        if orig.get("identical_files"):
            lines.append(f"🚨 Identical files (hard clone): `{'`, `'.join(orig['identical_files'])}`\n\n")

    # Improvement log
    lines.append("### Improvement Hypothesis\n\n")
    lines.append(entry["improvement_log"] + "\n\n")
    lines.append("---\n\n")

    with open(md_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"[evolution_tracker] Appended to EVOLUTION_LOG.md")
