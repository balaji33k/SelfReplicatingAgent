"""
analysis.py — Failure analysis with per-agent attribution.

Analyzes PipelineResult dicts produced by the multi-agent pipeline.
Tracks not just error_type but WHICH specialist agent was responsible,
enabling the EvolutionEngine to target specific agents in the next generation.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AnalysisReport:
    """
    Aggregated report of benchmark analysis.

    Fields added for multi-agent attribution:
      agent_failure_counts: how many tasks each agent contributed a failure to
      pipeline_metrics: cycle counts, agent usage frequency
    """

    def __init__(
        self,
        total_tasks: int,
        passed: int,
        failed: int,
        pass_rate: float,
        failure_breakdown: Dict[str, int],
        agent_failure_counts: Optional[Dict[str, int]] = None,
        pipeline_metrics: Optional[Dict[str, Any]] = None,
        sample_failures: Optional[List[Dict[str, Any]]] = None,
    ):
        self.total_tasks = total_tasks
        self.passed = passed
        self.failed = failed
        self.pass_rate = pass_rate
        self.failure_breakdown = failure_breakdown
        # Per-agent attribution: {"coder": 5, "critic": 2, ...}
        self.agent_failure_counts: Dict[str, int] = agent_failure_counts or {}
        # Pipeline efficiency: {"avg_revise_cycles": 1.2, ...}
        self.pipeline_metrics: Dict[str, Any] = pipeline_metrics or {}
        # Sample of failing tasks for LLM context
        self.sample_failures: List[Dict[str, Any]] = sample_failures or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "failure_breakdown": self.failure_breakdown,
            "agent_failure_counts": self.agent_failure_counts,
            "pipeline_metrics": self.pipeline_metrics,
            "sample_failures": self.sample_failures[:5],  # cap at 5 for compactness
        }

    def to_summary_string(self) -> str:
        """Human-readable summary for the EvolutionEngine prompt."""
        lines = [
            f"Total tasks attempted: {self.total_tasks}",
            f"Tasks passed: {self.passed}",
            f"Tasks failed: {self.failed}",
            f"Overall pass rate: {self.pass_rate:.2%}",
            "",
            "Error type breakdown:",
        ]
        for etype, count in (self.failure_breakdown or {}).items():
            lines.append(f"  {etype}: {count}")

        if self.agent_failure_counts:
            lines.append("")
            lines.append("Agent-level failure attribution:")
            for agent, count in sorted(
                self.agent_failure_counts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {agent}: contributed to {count} failure(s)")

        if self.pipeline_metrics:
            lines.append("")
            lines.append("Pipeline metrics:")
            for k, v in self.pipeline_metrics.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


class FailureAnalyzer:
    """
    Analyzes execution results (dict format from PipelineResult.to_dict()).

    Accepts both old-style ExecutionResult objects and new-style PipelineResult dicts.
    Extracts per-agent attribution from pipeline results.
    """

    def analyze_results(
        self, execution_results: List[Any], tasks: List[Any]
    ) -> AnalysisReport:
        """
        Aggregates results, classifies failures, attributes them to specialist agents.
        """
        if not execution_results:
            logger.warning("No execution results for analysis. Reporting zero pass rate.")
            return AnalysisReport(len(tasks), 0, len(tasks), 0.0, {})

        # Normalise to dicts
        result_dicts = []
        for r in execution_results:
            if isinstance(r, dict):
                result_dicts.append(r)
            elif hasattr(r, "to_dict"):
                result_dicts.append(r.to_dict())
            else:
                # Legacy ExecutionResult with direct attributes
                result_dicts.append({
                    "task_id": getattr(r, "task_id", "unknown"),
                    "success": getattr(r, "success", False),
                    "error_type": getattr(r, "error_type", "UnknownError"),
                    "stderr": getattr(r, "stderr", ""),
                    "stdout": getattr(r, "stdout", ""),
                    "runtime": getattr(r, "runtime", 0.0),
                    "agents_used": [],
                    "cycles": {},
                    "agent_failures": {},
                })

        attempted_ids = {r["task_id"] for r in result_dicts}
        failure_breakdown: Dict[str, int] = defaultdict(int)
        agent_failure_counts: Dict[str, int] = defaultdict(int)
        total_revise_cycles = 0
        total_debug_cycles = 0
        tasks_with_cycles = 0
        passed_count = 0
        sample_failures: List[Dict[str, Any]] = []

        # Tasks that never produced a result
        for task in tasks:
            tid = getattr(task, "task_id", None) or task.get("task_id", "?")
            if tid not in attempted_ids:
                failure_breakdown["SystemError_PreExecution"] += 1
                logger.warning(f"Task {tid} missing from results — assumed pre-execution failure")

        for r in result_dicts:
            if r.get("success"):
                passed_count += 1
            else:
                etype = r.get("error_type") or "UnknownError"
                failure_breakdown[etype] += 1

                # Per-agent attribution: blame the last agent that was active,
                # or use agent_failures dict if present
                agent_failures = r.get("agent_failures", {})
                if agent_failures:
                    for agent in agent_failures:
                        agent_failure_counts[agent] += 1
                else:
                    agents_used = r.get("agents_used", [])
                    if agents_used:
                        # Blame the last agent used
                        agent_failure_counts[agents_used[-1]] += 1

                if len(sample_failures) < 10:
                    sample_failures.append({
                        "task_id": r.get("task_id"),
                        "error_type": etype,
                        "stderr": (r.get("stderr") or "")[:200],
                        "agents_used": r.get("agents_used", []),
                        "agent_failures": agent_failures,
                    })

            # Accumulate cycle counts
            cycles = r.get("cycles", {})
            if cycles:
                rc = cycles.get("reviser", 0)
                dc = cycles.get("debugger", 0)
                if rc or dc:
                    total_revise_cycles += rc
                    total_debug_cycles += dc
                    tasks_with_cycles += 1

        final_total = len(tasks)
        final_passed = passed_count
        final_failed = final_total - final_passed
        pass_rate = final_passed / final_total if final_total > 0 else 0.0

        pipeline_metrics: Dict[str, Any] = {}
        if tasks_with_cycles:
            pipeline_metrics["avg_revise_cycles"] = round(
                total_revise_cycles / len(result_dicts), 2
            )
            pipeline_metrics["avg_debug_cycles"] = round(
                total_debug_cycles / len(result_dicts), 2
            )
            pipeline_metrics["tasks_that_needed_revision"] = tasks_with_cycles

        report = AnalysisReport(
            total_tasks=final_total,
            passed=final_passed,
            failed=final_failed,
            pass_rate=pass_rate,
            failure_breakdown=dict(failure_breakdown),
            agent_failure_counts=dict(agent_failure_counts),
            pipeline_metrics=pipeline_metrics,
            sample_failures=sample_failures,
        )
        logger.info(
            f"Analysis complete. Pass rate: {report.pass_rate:.2%} | "
            f"Agent attributions: {dict(agent_failure_counts)}"
        )
        return report
