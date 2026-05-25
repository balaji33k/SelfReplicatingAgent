"""
pipeline.py — Full LangGraph StateGraph pipeline.

Replaces manual orchestration with a compiled StateGraph.
AgentTopologyConfig (config.py) still drives which nodes are active —
enabling critic/reviser/debugger automatically adds cyclic edges.

Flow:
  START
    → analyst → architect → [test_writer?] → coder
    → [critic ⇄ reviser  (up to max_revise_cycles)]
    → executor
    → [debugger ⇄ executor (up to max_debug_cycles)]
    → END

Public interface is identical to the previous pipeline.py:
    pipeline = AgentPipeline(llm_client, topology_config)
    result   = pipeline.solve(task_contract, phase_callback=cb)
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from contracts import (
    AnalysisSpec, CodeArtifact, CritiqueReport, DesignSpec,
    ExecutionContract, PipelineResult, TaskContract, TestSuite,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. SHARED STATE
#    Single TypedDict flows through every node.
#    Each node returns only the fields it updated — LangGraph merges.
# ══════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict, total=False):
    task:            TaskContract
    analysis:        Optional[AnalysisSpec]
    design:          Optional[DesignSpec]
    extra_tests:     Optional[TestSuite]
    artifact:        Optional[CodeArtifact]
    critique:        Optional[CritiqueReport]
    execution:       Optional[ExecutionContract]
    revise_count:    int
    debug_count:     int
    agents_used:     List[str]
    phase_results:   Dict[str, Optional[bool]]
    agent_failures:  Dict[str, str]


# ══════════════════════════════════════════════════════════════════════
# 2. NODE FACTORIES
#    Each factory closes over its agent instance.
#    Returns a plain (state) → dict function — LangGraph calls it.
# ══════════════════════════════════════════════════════════════════════

def _make_analyst_node(agent):
    def node(state: PipelineState) -> dict:
        logger.info("[pipeline] analyst START")
        return {
            "analysis":   agent.run(state["task"]),
            "agents_used": state.get("agents_used", []) + ["analyst"],
        }
    return node


def _make_architect_node(agent):
    def node(state: PipelineState) -> dict:
        logger.info("[pipeline] architect START")
        return {
            "design":     agent.run(state["analysis"]),
            "agents_used": state.get("agents_used", []) + ["architect"],
        }
    return node


def _make_test_writer_node(agent):
    def node(state: PipelineState) -> dict:
        logger.info("[pipeline] test_writer START")
        try:
            # Information isolation: TestWriter sees AnalysisSpec ONLY
            extra = agent.run(state["analysis"])
            return {
                "extra_tests":  extra,
                "agents_used":  state.get("agents_used", []) + ["test_writer"],
            }
        except Exception as e:
            logger.warning(f"[pipeline] test_writer failed (non-fatal): {e}")
            failures = dict(state.get("agent_failures", {}))
            failures["test_writer"] = str(e)
            return {"agent_failures": failures}
    return node


def _make_coder_node(agent):
    def node(state: PipelineState) -> dict:
        logger.info("[pipeline] coder START")
        return {
            "artifact":   agent.run(state["design"], state["analysis"]),
            "agents_used": state.get("agents_used", []) + ["coder"],
        }
    return node


def _make_critic_node(agent):
    def node(state: PipelineState) -> dict:
        logger.info("[pipeline] critic START")
        critique = agent.run(state["artifact"], state["analysis"])
        used = state.get("agents_used", [])
        if "critic" not in used:
            used = used + ["critic"]
        return {"critique": critique, "agents_used": used}
    return node


def _make_reviser_node(agent):
    def node(state: PipelineState) -> dict:
        cycle = state.get("revise_count", 0) + 1
        logger.info(f"[pipeline] reviser START cycle={cycle}")
        revised = agent.run(state["artifact"], state["critique"], cycle=cycle)
        used = state.get("agents_used", [])
        if "reviser" not in used:
            used = used + ["reviser"]
        return {
            "artifact":    revised,
            "revise_count": cycle,
            "agents_used": used,
        }
    return node


def _executor_node(state: PipelineState) -> dict:
    """Run generated code in a subprocess sandbox — no LLM call."""
    logger.info("[pipeline] executor START")
    task     = state["task"]
    artifact = state["artifact"]
    extra    = state.get("extra_tests")

    all_tests = list(task.test_cases)
    if extra and extra.extra_tests:
        all_tests.extend(extra.extra_tests)

    execution = _run_in_sandbox(artifact, all_tests)

    stderr_lower = (execution.stderr or "").lower()
    compile_ok = not any(k in stderr_lower for k in
                         ("syntaxerror", "importerror", "modulenotfounderror"))
    phase_results = dict(state.get("phase_results", {}))
    phase_results["compile"]    = compile_ok
    phase_results["unit_tests"] = execution.success

    used = state.get("agents_used", [])
    if "executor" not in used:
        used = used + ["executor"]

    return {
        "execution":     execution,
        "phase_results": phase_results,
        "agents_used":   used,
    }


def _make_debugger_node(agent):
    def node(state: PipelineState) -> dict:
        cycle = state.get("debug_count", 0) + 1
        logger.info(f"[pipeline] debugger START cycle={cycle}")
        patched = agent.run(state["artifact"], state["execution"])
        used = state.get("agents_used", [])
        if "debugger" not in used:
            used = used + ["debugger"]
        return {
            "artifact":   patched,
            "debug_count": cycle,
            "agents_used": used,
        }
    return node


# ══════════════════════════════════════════════════════════════════════
# 3. CONDITIONAL EDGE ROUTERS
#    Return the name of the next node as a string.
#    Replaces the manual loop logic in the old pipeline.py.
# ══════════════════════════════════════════════════════════════════════

def _make_critic_router(max_cycles: int):
    def router(state: PipelineState) -> str:
        critique = state.get("critique")
        count    = state.get("revise_count", 0)
        if critique and not critique.passed and count < max_cycles:
            logger.info(f"[pipeline] critic → reviser (cycle {count+1}/{max_cycles})")
            return "reviser"
        logger.info("[pipeline] critic → executor")
        return "executor"
    return router


def _make_executor_router(max_cycles: int, has_debugger: bool):
    def router(state: PipelineState) -> str:
        execution = state.get("execution")
        count     = state.get("debug_count", 0)
        if has_debugger and execution and not execution.success and count < max_cycles:
            logger.info(f"[pipeline] executor → debugger (cycle {count+1}/{max_cycles})")
            return "debugger"
        return END
    return router


# ══════════════════════════════════════════════════════════════════════
# 4. GRAPH BUILDER
#    Builds and compiles a StateGraph from topology config.
#    Called once at AgentPipeline.__init__.
# ══════════════════════════════════════════════════════════════════════

def _build_graph(agents: dict, cfg):
    g = StateGraph(PipelineState)

    # ── Always-on nodes ─────────────────────────────────────────────
    g.add_node("analyst",   _make_analyst_node(agents["analyst"]))
    g.add_node("architect", _make_architect_node(agents["architect"]))
    g.add_node("coder",     _make_coder_node(agents["coder"]))
    g.add_node("executor",  _executor_node)

    # ── Optional: TestWriter ─────────────────────────────────────────
    if cfg.enable_test_writer and "test_writer" in agents:
        g.add_node("test_writer", _make_test_writer_node(agents["test_writer"]))
        g.add_edge("architect",  "test_writer")
        g.add_edge("test_writer", "coder")
    else:
        g.add_edge("architect", "coder")

    # ── Optional: Critic + Reviser cycle ────────────────────────────
    if cfg.enable_critic and "critic" in agents:
        g.add_node("critic", _make_critic_node(agents["critic"]))
        g.add_edge("coder", "critic")

        if cfg.enable_reviser and "reviser" in agents:
            g.add_node("reviser", _make_reviser_node(agents["reviser"]))
            g.add_edge("reviser", "critic")          # ← cycle back to critic

        g.add_conditional_edges(
            "critic",
            _make_critic_router(cfg.max_revise_cycles),
            {"reviser": "reviser", "executor": "executor"},
        )
    else:
        g.add_edge("coder", "executor")

    # ── Optional: Debugger cycle ─────────────────────────────────────
    if cfg.enable_debugger and "debugger" in agents:
        g.add_node("debugger", _make_debugger_node(agents["debugger"]))
        g.add_edge("debugger", "executor")           # ← re-execute after patch
        g.add_conditional_edges(
            "executor",
            _make_executor_router(cfg.max_debug_cycles, has_debugger=True),
            {"debugger": "debugger", END: END},
        )
    else:
        g.add_conditional_edges(
            "executor",
            _make_executor_router(cfg.max_debug_cycles, has_debugger=False),
            {END: END},
        )

    g.add_edge(START, "analyst")
    return g.compile()


# ══════════════════════════════════════════════════════════════════════
# 5. PUBLIC AgentPipeline
#    Drop-in replacement — main.py is unchanged.
# ══════════════════════════════════════════════════════════════════════

class AgentPipeline:
    """LangGraph-backed pipeline. Same interface as the previous pipeline.py."""

    # Node name → dashboard phase label
    _PHASE_MAP = {
        "analyst":    "thinking",
        "architect":  "thinking",
        "test_writer":"writing_unit_tests",
        "coder":      "writing_code",
        "critic":     "reviewing",
        "reviser":    "revising",
        "executor":   "compiling",
        "debugger":   "debugging",
    }

    def __init__(self, llm_client, topology_config):
        self.cfg     = topology_config
        self._agents = self._build_agents(llm_client)
        self._graph  = _build_graph(self._agents, topology_config)
        logger.info(
            f"[pipeline] LangGraph compiled — agents: {list(self._agents.keys())} | "
            f"critic={topology_config.enable_critic} "
            f"reviser={topology_config.enable_reviser} "
            f"test_writer={topology_config.enable_test_writer} "
            f"debugger={topology_config.enable_debugger}"
        )

    def _build_agents(self, llm_client) -> dict:
        from agent_analyst     import AnalystAgent
        from agent_architect   import ArchitectAgent
        from agent_coder       import CoderAgent
        from agent_critic      import CriticAgent
        from agent_reviser     import ReviserAgent
        from agent_test_writer import TestWriterAgent
        from agent_debugger    import DebuggerAgent

        cfg = self.cfg
        agents = {
            "analyst":   AnalystAgent(llm_client),
            "architect": ArchitectAgent(llm_client),
            "coder":     CoderAgent(llm_client),
        }
        if cfg.enable_critic:      agents["critic"]      = CriticAgent(llm_client)
        if cfg.enable_reviser:     agents["reviser"]     = ReviserAgent(llm_client)
        if cfg.enable_test_writer: agents["test_writer"] = TestWriterAgent(llm_client)
        if cfg.enable_debugger:    agents["debugger"]    = DebuggerAgent(llm_client)
        return agents

    def solve(self, task: TaskContract, phase_callback=None) -> PipelineResult:
        """Run the compiled graph for one task. Returns PipelineResult."""
        start_time = time.monotonic()
        initial_state: PipelineState = {
            "task":          task,
            "revise_count":  0,
            "debug_count":   0,
            "agents_used":   [],
            "phase_results": {},
            "agent_failures":{},
        }

        try:
            final_state: dict = {}

            # stream_mode="updates" → yields {node_name: {changed_fields}}
            # after each node completes — drives live dashboard phase callbacks.
            for chunk in self._graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    final_state.update(node_update)
                    if phase_callback:
                        phase = self._PHASE_MAP.get(node_name, node_name)
                        try:
                            phase_callback(phase, node_update.get("phase_results", {}))
                        except Exception:
                            pass

            execution: ExecutionContract = final_state.get("execution")
            artifact:  CodeArtifact     = final_state.get("artifact")
            elapsed = time.monotonic() - start_time

            return PipelineResult(
                task_id      = task.task_id,
                success      = bool(execution and execution.success),
                final_code   = artifact.code if artifact else "",
                error_type   = None if (execution and execution.success)
                               else (execution.error_type if execution else "NoExecution"),
                runtime      = elapsed,
                stdout       = execution.stdout if execution else "",
                stderr       = execution.stderr if execution else "",
                agents_used  = final_state.get("agents_used",   []),
                cycles       = {
                    "reviser":  final_state.get("revise_count", 0),
                    "debugger": final_state.get("debug_count",  0),
                },
                agent_failures = final_state.get("agent_failures", {}),
                phase_results  = final_state.get("phase_results",  {}),
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"[pipeline] Pipeline crashed for task {task.task_id}: {e}")
            return PipelineResult(
                task_id=task.task_id, success=False, final_code="",
                error_type=type(e).__name__, runtime=elapsed,
                stdout="", stderr=str(e),
            )


# ══════════════════════════════════════════════════════════════════════
# 6. SANDBOX EXECUTOR (no LLM — unchanged logic from old pipeline.py)
# ══════════════════════════════════════════════════════════════════════

def _run_in_sandbox(
    artifact: CodeArtifact,
    test_cases: list,
    timeout: int = 10,
) -> ExecutionContract:
    lines = ["import sys, json", f"\n{artifact.code}\n", "results=[]"]

    for tc in test_cases:
        raw = tc.get("input")
        exp = tc.get("expected")
        if isinstance(raw, dict):
            call = f"{artifact.function_name}(**{raw!r})"
        elif isinstance(raw, list):
            call = f"{artifact.function_name}(*{raw!r})"
        else:
            call = f"{artifact.function_name}({raw!r})"
        lines.append(
            f"try:\n"
            f"    _r={call}\n"
            f"    results.append({{'input':{raw!r},'expected':{exp!r},'actual':_r,'passed':_r=={exp!r}}})\n"
            f"except Exception as _e:\n"
            f"    results.append({{'input':{raw!r},'expected':{exp!r},'actual':None,'passed':False,'error':str(_e)}})"
        )
    lines.append("print(json.dumps(results))")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                return ExecutionContract(
                    artifact.task_id, False,
                    proc.stdout, proc.stderr, "RuntimeError", 0.0,
                )
            import json as _j
            results = _j.loads(proc.stdout) if proc.stdout.startswith("[") else []
            ok = bool(results) and all(r.get("passed") for r in results)
            return ExecutionContract(
                artifact.task_id, ok,
                proc.stdout, proc.stderr,
                None if ok else "AssertionError", 0.0, results,
            )
        except subprocess.TimeoutExpired:
            return ExecutionContract(
                artifact.task_id, False, "", "Timeout", "TimeoutError", float(timeout),
            )
        except Exception as e:
            return ExecutionContract(
                artifact.task_id, False, "", str(e), type(e).__name__, 0.0,
            )
