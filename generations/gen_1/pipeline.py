"""
pipeline.py — AgentPipeline orchestrator.

Drives the specialist agents in sequence according to the AgentTopologyConfig.
The pipeline is data-driven: topology changes propagate through config.py,
not through hardcoded if/else chains.

Flow:
  TaskContract
     ↓ Analyst
  AnalysisSpec
     ↓ Architect           ↘ TestWriter (parallel, from AnalysisSpec only)
  DesignSpec              TestSuite
     ↓ Coder
  CodeArtifact
     ↓ Critic (uses AnalysisSpec + CodeArtifact — NOT DesignSpec)
  CritiqueReport
     ↓ Reviser (if critique.passed == False, up to max_revise_cycles)
  CodeArtifact (revised)
     ↓ Executor
  ExecutionContract
     ↓ Debugger (if execution.success == False)
  CodeArtifact (patched)
     ↓ Final execution
  PipelineResult
"""
import logging
import subprocess
import sys
import tempfile
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from contracts import (
    AnalysisSpec,
    CodeArtifact,
    CritiqueReport,
    DesignSpec,
    ExecutionContract,
    PipelineResult,
    TaskContract,
    TestSuite,
)

logger = logging.getLogger(__name__)


class AgentPipeline:
    """
    Orchestrates the 7-agent specialization pipeline.
    Each agent is only constructed when it is enabled in the topology config.

    Information isolation is enforced here:
      - TestWriter receives AnalysisSpec ONLY (not CodeArtifact or DesignSpec)
      - Critic receives CodeArtifact + AnalysisSpec ONLY (not DesignSpec)
      - Debugger receives CodeArtifact + ExecutionContract ONLY
    """

    def __init__(self, llm_client, topology_config):
        """
        Args:
            llm_client: LLMClient instance
            topology_config: AgentTopologyConfig from config.py
        """
        self.llm = llm_client
        self.cfg = topology_config
        self._agents: Dict[str, Any] = {}
        self._build_agents()

    def _build_agents(self) -> None:
        """Construct only the agents that are enabled in the topology config."""
        from agent_analyst import AnalystAgent
        from agent_architect import ArchitectAgent
        from agent_coder import CoderAgent
        from agent_critic import CriticAgent
        from agent_reviser import ReviserAgent
        from agent_test_writer import TestWriterAgent
        from agent_debugger import DebuggerAgent

        # These 3 are always required
        self._agents["analyst"] = AnalystAgent(self.llm)
        self._agents["architect"] = ArchitectAgent(self.llm)
        self._agents["coder"] = CoderAgent(self.llm)

        # Optional agents controlled by topology
        if self.cfg.enable_critic:
            self._agents["critic"] = CriticAgent(self.llm)
        if self.cfg.enable_reviser:
            self._agents["reviser"] = ReviserAgent(self.llm)
        if self.cfg.enable_test_writer:
            self._agents["test_writer"] = TestWriterAgent(self.llm)
        if self.cfg.enable_debugger:
            self._agents["debugger"] = DebuggerAgent(self.llm)

        logger.info(f"[pipeline] Agents built: {list(self._agents.keys())}")

    # ── Public entry point ────────────────────────────────────────────────────

    def solve(self, task: TaskContract, phase_callback=None) -> PipelineResult:
        """
        Run the full pipeline for a single task.
        Returns a PipelineResult with full agent provenance.

        phase_callback(phase_name: str, phase_results: dict) is called before
        each major stage so the dashboard can show live activity.
        Phase names: thinking | writing_unit_tests | writing_code | reviewing |
                     revising | compiling | running_unit_tests |
                     debugging | running_integration_tests
        """
        agents_used: List[str] = []
        cycles: Dict[str, int] = {"reviser": 0, "debugger": 0}
        agent_failures: Dict[str, str] = {}
        phase_results: Dict = {}   # compile / unit_tests / integration_tests
        start_time = time.monotonic()

        def _fire(phase: str, ph: dict = None):
            """Fire phase callback, silently swallowing any errors."""
            if phase_callback:
                try:
                    phase_callback(phase, ph or {})
                except Exception:
                    pass

        try:
            # ── Stage 1: Analyst ────────────────────────────────────────────────
            _fire("thinking")
            agents_used.append("analyst")
            analysis: AnalysisSpec = self._agents["analyst"].run(task)

            # ── Stage 2: Architect ──────────────────────────────────────────────
            agents_used.append("architect")
            design: DesignSpec = self._agents["architect"].run(analysis)

            # ── Stage 2b: TestWriter (parallel path — from AnalysisSpec ONLY) ──
            extra_tests: Optional[TestSuite] = None
            if "test_writer" in self._agents:
                _fire("writing_unit_tests")
                agents_used.append("test_writer")
                try:
                    extra_tests = self._agents["test_writer"].run(analysis)
                    logger.info(
                        f"[pipeline] TestWriter generated {len(extra_tests.extra_tests)} extra tests"
                    )
                except Exception as e:
                    agent_failures["test_writer"] = str(e)
                    logger.warning(f"[pipeline] TestWriter failed (non-fatal): {e}")

            # ── Stage 3: Coder ──────────────────────────────────────────────────
            _fire("writing_code")
            agents_used.append("coder")
            artifact: CodeArtifact = self._agents["coder"].run(design, analysis)

            # ── Stage 4: Critic → Reviser loop ─────────────────────────────────
            if "critic" in self._agents:
                _fire("reviewing")
                artifact, cycles["reviser"], reviser_failures = self._critic_reviser_loop(
                    artifact, analysis, _fire
                )
                agents_used.append("critic")
                if cycles["reviser"] > 0:
                    agents_used.append("reviser")
                if reviser_failures:
                    agent_failures["reviser"] = reviser_failures

            # ── Stage 5: Merge extra tests with task tests ───────────────────────
            all_tests = list(task.test_cases)
            if extra_tests and extra_tests.extra_tests:
                all_tests.extend(extra_tests.extra_tests)

            # ── Stage 6: Execute (first run — compile + unit tests) ──────────────
            _fire("compiling")
            agents_used.append("executor")
            execution = self._execute(artifact, all_tests)

            # Classify compile vs logic failures
            stderr_lower = (execution.stderr or "").lower()
            compile_ok = (
                "syntaxerror" not in stderr_lower
                and "importerror" not in stderr_lower
                and "modulenotfounderror" not in stderr_lower
            )
            phase_results["compile"] = compile_ok
            unit_ok = execution.success
            phase_results["unit_tests"] = unit_ok
            _fire("running_unit_tests", dict(phase_results))

            # ── Stage 7: Debugger (on failure) ──────────────────────────────────
            if not execution.success and "debugger" in self._agents:
                _fire("debugging", dict(phase_results))
                agents_used.append("debugger")
                artifact, cycles["debugger"] = self._debug_loop(artifact, execution)
                _fire("running_integration_tests", dict(phase_results))
                # Re-execute after debugging
                execution = self._execute(artifact, all_tests)
                integ_ok = execution.success
                phase_results["integration_tests"] = integ_ok

            elapsed = time.monotonic() - start_time
            return PipelineResult(
                task_id=task.task_id,
                success=execution.success,
                final_code=artifact.code,
                error_type=None if execution.success else execution.error_type,
                runtime=elapsed,
                stdout=execution.stdout,
                stderr=execution.stderr,
                agents_used=agents_used,
                cycles=cycles,
                agent_failures=agent_failures,
                phase_results=phase_results,
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"[pipeline] Pipeline crashed for task {task.task_id}: {e}")
            return PipelineResult(
                task_id=task.task_id,
                success=False,
                final_code="",
                error_type=type(e).__name__,
                runtime=elapsed,
                stdout="",
                stderr=str(e),
                agents_used=agents_used,
                cycles=cycles,
                agent_failures=agent_failures,
                phase_results=phase_results,
            )

    # ── Critic → Reviser loop ─────────────────────────────────────────────────

    def _critic_reviser_loop(
        self, artifact: CodeArtifact, analysis: AnalysisSpec, fire_fn=None
    ):
        """
        Run Critic, then Reviser up to max_revise_cycles.
        Returns (final_artifact, revision_count, failure_summary).
        fire_fn: optional callable(phase, phase_results) from the parent solve().
        """
        revisions = 0
        failures = ""

        for cycle in range(1, self.cfg.max_revise_cycles + 1):
            critique: CritiqueReport = self._agents["critic"].run(artifact, analysis)

            if critique.passed:
                logger.info(f"[pipeline] Critic PASS on cycle {cycle} — skipping revision")
                break

            logger.info(
                f"[pipeline] Critic FAIL (severity={critique.severity}) — "
                f"cycle {cycle}/{self.cfg.max_revise_cycles}"
            )

            if "reviser" not in self._agents:
                failures = "critic_failed_no_reviser"
                break

            if fire_fn:
                try:
                    fire_fn("revising", {})
                except Exception:
                    pass

            revised = self._agents["reviser"].run(artifact, critique, cycle=cycle)
            revisions += 1

            # If code didn't change, stop cycling
            if revised.code.strip() == artifact.code.strip():
                logger.warning(f"[pipeline] Reviser produced identical code — stopping")
                failures = f"reviser_stalled_at_cycle_{cycle}"
                break

            artifact = revised

        return artifact, revisions, failures

    # ── Debugger loop ─────────────────────────────────────────────────────────

    def _debug_loop(self, artifact: CodeArtifact, execution: ExecutionContract):
        """
        Run Debugger up to max_debug_cycles.
        Returns (final_artifact, debug_count).
        """
        debug_count = 0

        for cycle in range(1, self.cfg.max_debug_cycles + 1):
            patched = self._agents["debugger"].run(artifact, execution)
            debug_count += 1

            # Debugger flagged as needs redesign
            if "::needs_redesign" in patched.task_id:
                logger.warning(
                    f"[pipeline] Debugger flagged NEEDS REDESIGN — stopping debug loop"
                )
                # Return original (un-patched) — let final execution surface the failure
                return artifact, debug_count

            if patched.code.strip() == artifact.code.strip():
                logger.warning(
                    f"[pipeline] Debugger returned identical code — stopping at cycle {cycle}"
                )
                break

            artifact = patched
            break  # One debug cycle per execution failure

        return artifact, debug_count

    # ── Executor ──────────────────────────────────────────────────────────────

    def _execute(self, artifact: CodeArtifact, test_cases: List[Dict]) -> ExecutionContract:
        """
        Execute the code in a subprocess sandbox against the test cases.
        Mirrors sandbox_execution.py logic but returns ExecutionContract.
        """
        timeout = getattr(self.cfg, "execution_timeout", 10)

        # Build test harness script
        test_lines = [
            "import sys, json",
            f"\n{artifact.code}\n",
            "results = []",
        ]

        for tc in test_cases:
            raw_input = tc.get("input")
            expected = tc.get("expected")
            # Dict input → keyword args: func(**{"nums": [1,2], "target": 3})
            # List input → positional args: func(*[1, 2, 3])
            # Scalar input → single arg:   func(42)
            if isinstance(raw_input, dict):
                call_expr = f"{artifact.function_name}(**{raw_input!r})"
                inp_repr = raw_input
            elif isinstance(raw_input, list):
                call_expr = f"{artifact.function_name}(*{raw_input!r})"
                inp_repr = raw_input
            else:
                call_expr = f"{artifact.function_name}({raw_input!r})"
                inp_repr = raw_input
            test_lines.append(
                f"try:\n"
                f"    result = {call_expr}\n"
                f"    passed = (result == {expected!r})\n"
                f"    results.append({{'input': {inp_repr!r}, 'expected': {expected!r}, "
                f"'actual': result, 'passed': passed}})\n"
                f"except Exception as _e:\n"
                f"    results.append({{'input': {inp_repr!r}, 'expected': {expected!r}, "
                f"'actual': None, 'passed': False, 'error': str(_e)}})"
            )

        test_lines.append("print(json.dumps(results))")
        script = "\n".join(test_lines)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "solution.py")
            with open(script_path, "w") as f:
                f.write(script)

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if proc.returncode != 0:
                    return ExecutionContract(
                        task_id=artifact.task_id,
                        success=False,
                        stdout=stdout,
                        stderr=stderr,
                        error_type="RuntimeError",
                        runtime=0.0,
                        test_results=[],
                    )

                import json as _json
                test_results = _json.loads(stdout) if stdout.startswith("[") else []
                all_passed = bool(test_results) and all(r.get("passed") for r in test_results)

                return ExecutionContract(
                    task_id=artifact.task_id,
                    success=all_passed,
                    stdout=stdout,
                    stderr=stderr,
                    error_type=None if all_passed else "AssertionError",
                    runtime=0.0,
                    test_results=test_results,
                )

            except subprocess.TimeoutExpired:
                return ExecutionContract(
                    task_id=artifact.task_id,
                    success=False,
                    stdout="",
                    stderr="Timeout",
                    error_type="TimeoutError",
                    runtime=float(timeout),
                    test_results=[],
                )
            except Exception as e:
                return ExecutionContract(
                    task_id=artifact.task_id,
                    success=False,
                    stdout="",
                    stderr=str(e),
                    error_type=type(e).__name__,
                    runtime=0.0,
                    test_results=[],
                )
