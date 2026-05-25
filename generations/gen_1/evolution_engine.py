"""
evolution_engine.py — Designs the next generation from failure evidence.

Groq-compatible design: generates files ONE AT A TIME to stay within the
12,000 tokens-per-minute free-tier limit. Each call generates a single .py
file (~2,000 tokens) rather than all 18 files at once (~50,000 tokens).

Key guarantees:
  1. The LLM receives the full failure analysis for every file it generates.
  2. Each agent file gets a targeted prompt: "fix the failures this agent caused".
  3. Anti-clone: every file must differ meaningfully from its parent version.
  4. Missing or compile-invalid files abort the spawn (no silent failures).
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from analysis import AnalysisReport
from config import Config, LLMConfig
from llm_client import LLMClient

logger = logging.getLogger(__name__)

# Infrastructure files the LLM should NOT regenerate
INFRA_FILES = {"telemetry.py", "lineage_memory.py"}

# Files the LLM MUST generate (one at a time on Groq free tier).
# NOTE: llm_client.py is NOT listed here — it is infrastructure (ChatGroq-backed)
# that spawner.py copies from the parent generation to preserve provider failover.
REQUIRED_FILES = [
    "main.py", "config.py", "task_manager.py",
    "analysis.py", "evolution_engine.py", "spawner.py",
    # Multi-agent pipeline (LangGraph StateGraph)
    "contracts.py", "agent_base.py", "pipeline.py",
    # Specialist agents
    "agent_analyst.py", "agent_architect.py", "agent_coder.py",
    "agent_critic.py", "agent_reviser.py", "agent_test_writer.py",
    "agent_debugger.py",
    # Clone guard
    "agent_clone_inspector.py",
]

# Delay between per-file LLM calls to stay inside TPM budget (seconds)
# 12K TPM / ~2.5K tokens per call = 4 calls/min safely with 15s gap
INTER_FILE_DELAY = 15


@dataclass
class NextGenerationDesign:
    """Holds the complete design for the next generation."""
    new_config: Config
    new_prompts_content: Dict[str, str] = field(default_factory=dict)
    improvement_log_message: str = ""


class EvolutionEngine:
    """
    Designs the next generation one file at a time.

    Two-phase approach:
      Phase 1 — Planning (~500 tokens): asks the LLM for a concise improvement
                plan that explains what to change in each file and why.
      Phase 2 — Generation (one call per file, ~2K tokens each): uses the plan
                to generate each file individually with a targeted prompt.

    This keeps every API call under ~3,000 tokens, well within Groq's
    12,000 TPM free-tier limit.
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_client = LLMClient(llm_config)
        self.gen_dir = Path(__file__).resolve().parent

    # ── Public API ────────────────────────────────────────────────────────────

    def design_next_generation(
        self, analysis_report: AnalysisReport, parent_config: Config
    ) -> NextGenerationDesign:
        gen_num = parent_config.generation_number
        next_gen = gen_num + 1
        logger.info(f"Designing Generation {next_gen} from failure evidence (one file at a time)...")

        failure_ctx = self._build_failure_context(analysis_report, parent_config)
        current_source = self._load_current_source()

        # Phase 1: Get improvement plan
        improvement_log = self._get_improvement_plan(failure_ctx, current_source, gen_num, next_gen)
        logger.info(f"Improvement plan: {improvement_log[:120]}...")

        # Phase 2: Generate each file individually
        files: Dict[str, str] = {}
        for i, fname in enumerate(REQUIRED_FILES):
            logger.info(f"  Generating [{i+1}/{len(REQUIRED_FILES)}]: {fname}")
            parent_snippet = current_source.get(fname, "")[:400]
            content = self._generate_one_file(
                fname=fname,
                improvement_log=improvement_log,
                failure_ctx=failure_ctx,
                parent_snippet=parent_snippet,
                gen_num=gen_num,
                next_gen=next_gen,
            )
            files[fname] = content
            # Respect TPM budget — wait between calls (except after the last)
            if i < len(REQUIRED_FILES) - 1:
                time.sleep(INTER_FILE_DELAY)

        missing = [f for f in REQUIRED_FILES if f not in files]
        if missing:
            raise RuntimeError(f"Evolution incomplete — missing files: {missing}")

        logger.info(f"Generation {next_gen} design complete. {len(files)} files generated.")

        new_config = parent_config
        new_config.improvement_log = improvement_log
        return NextGenerationDesign(
            new_config=new_config,
            new_prompts_content=files,
            improvement_log_message=improvement_log,
        )

    # ── Phase 1: Planning ─────────────────────────────────────────────────────

    def _get_improvement_plan(
        self,
        failure_ctx: str,
        current_source: Dict[str, str],
        gen_num: int,
        next_gen: int,
    ) -> str:
        """
        Ask the LLM for a concise improvement plan (plain text, no code).
        Kept short so it fits in ~500 output tokens.
        """
        source_summary = "\n".join(
            f"- {fname}: {len(src)} chars" for fname, src in current_source.items()
        )
        prompt = f"""You are designing Generation {next_gen} of a self-replicating AI coding agent.

AGGREGATE FAILURE STATISTICS (Generation {gen_num}):
{failure_ctx}

CURRENT ARCHITECTURE FILES:
{source_summary}

⚠️  GENERAL IMPROVEMENTS ONLY — DO NOT make task-specific fixes.
    The agent must improve at solving ANY coding problem, not patch specific test cases.
    Treat the failure statistics as signals about SYSTEMIC weaknesses in the agent design.

    Error type → architectural root cause:
      AssertionError  → reasoning/logic gap in Coder agent prompt (add chain-of-thought)
      SyntaxError     → markdown fences not stripped, or Coder generating prose instead of code
      ImportError     → Coder hallucinates modules; prompt must restrict to stdlib + task-given libs
      TimeoutError    → Coder generates O(n²)+ algorithms; prompt must ask for complexity analysis
      RuntimeError    → edge cases unhandled; Critic/Debugger cycle may need enabling

Write a concise improvement plan (plain text, no code, max 300 words) that explains:
1. What SYSTEMIC weaknesses the error-type distribution reveals
2. Which architectural files need to change (agent prompts, topology config, pipeline logic)
3. What general pass-rate improvement is expected in Generation {next_gen}

ARCHITECTURE CONSTRAINTS (must be preserved):
- Pipeline uses LangGraph StateGraph (langgraph>=0.2.0); do NOT revert to custom loops
- LLM client uses ChatGroq (langchain-groq) — llm_client.py is infra, copied automatically
- Model: meta-llama/llama-4-scout-17b-16e-instruct on Groq (30k TPM / 500k TPD)"""

        response = self.llm_client.call(prompt)
        return response.strip()[:1500]  # Cap at 1500 chars

    # ── Phase 2: Per-file generation ──────────────────────────────────────────

    def _generate_one_file(
        self,
        fname: str,
        improvement_log: str,
        failure_ctx: str,
        parent_snippet: str,
        gen_num: int,
        next_gen: int,
    ) -> str:
        """
        Generate a single .py file for the next generation.
        The prompt is deliberately compact to stay within Groq's TPM budget.
        """
        anti_clone = (
            f"⚠️ ANTI-CLONE: Your output must differ meaningfully from the parent version. "
            f"The spawner performs a byte-for-byte check — identical files abort the spawn."
        )

        file_role = self._file_role(fname)

        prompt = f"""You are generating {fname} for Generation {next_gen} of a self-replicating AI coding agent.

{anti_clone}

IMPROVEMENT PLAN (architectural changes and why):
{improvement_log[:600]}

AGGREGATE FAILURE STATISTICS (use as design signals — no task-specific fixes):
{failure_ctx[:500]}

FILE ROLE: {file_role}

PARENT VERSION (first 400 chars — DO NOT COPY, only use as structural reference):
{parent_snippet}

⚠️  GENERAL IMPROVEMENTS ONLY:
    This file must become ARCHITECTURALLY better — improve agent prompts, reasoning
    strategies, pipeline topology, or error handling in ways that help solve ANY
    coding task. Do NOT hardcode fixes for specific task inputs or expected outputs.

REQUIREMENTS:
- File: {fname}  Generation: {next_gen}
- Address the SYSTEMIC failure patterns shown in the statistics above
- llm_client.py is INFRASTRUCTURE (ChatGroq-backed) — do NOT generate it; it is copied automatically
- LLMClient uses ChatGroq (langchain-groq), reads GROQ_API_KEY from env, max_retries=7
- Default Groq model: meta-llama/llama-4-scout-17b-16e-instruct (30k TPM / 500k TPD)
- pipeline.py MUST use LangGraph (langgraph.graph.StateGraph) — NOT custom loops
  • Import: from langgraph.graph import StateGraph, START, END
  • PipelineState as TypedDict flows through all nodes
  • Conditional edges replace _critic_reviser_loop / _debug_loop methods
  • Compile graph once in __init__; stream with stream_mode="updates" in solve()
- Every agent file must own its prompt internally (not rely on external prompt files)
- Keep the same data contracts (TaskContract, AnalysisSpec, CodeArtifact, etc.)
- requirements.txt already includes: langgraph>=0.2.0, langchain-groq>=0.2.0, langchain-core>=0.3.0

OUTPUT: Write ONLY the complete Python source code for {fname}.
Start with the module docstring. No explanation outside the code."""

        response = self.llm_client.call(prompt)
        return self._extract_code(response, fname)

    def _file_role(self, fname: str) -> str:
        """One-line description of what each file is responsible for."""
        roles = {
            "main.py": "Entry point — loads tasks, runs pipeline, triggers evolution",
            "config.py": "Configuration — AgentTopologyConfig, LLMConfig, model defaults",
            "task_manager.py": "Task loading from problem_pool.json",
            "analysis.py": "Failure analysis — classifies errors, attributes to agents",
            "evolution_engine.py": "Evolution — designs next generation from failure evidence",
            "spawner.py": "Spawner — validates, writes, and launches the next generation",
            "contracts.py": "Data contracts — typed dataclasses between agents",
            "agent_base.py": "Base class for all specialist agents",
            "pipeline.py": "AgentPipeline — LangGraph StateGraph (langgraph>=0.2.0); nodes per AgentTopologyConfig; PipelineState TypedDict; conditional edges for critic/debug cycles; stream_mode='updates'",
            "agent_analyst.py": "Analyst agent — decomposes problem into AnalysisSpec",
            "agent_architect.py": "Architect agent — designs solution as DesignSpec",
            "agent_coder.py": "Coder agent — generates working Python code; use chain-of-thought",
            "agent_critic.py": "Critic agent — reviews code against spec (not design intent)",
            "agent_reviser.py": "Reviser agent — applies Critic's targeted fixes only",
            "agent_test_writer.py": "TestWriter agent — generates tests from AnalysisSpec ONLY",
            "agent_debugger.py": "Debugger agent — patches exact failure, no redesign",
            "agent_clone_inspector.py": "Clone Inspector — detects if offspring files copy parent",
        }
        return roles.get(fname, f"Logic file: {fname}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_failure_context(self, analysis_report: AnalysisReport, parent_config: Config) -> str:
        """
        Compact, TASK-AGNOSTIC failure summary for every per-file prompt.

        Intentionally omits task names, problem statements, and sample I/O —
        those are task-specific details that would cause the LLM to hardcode
        fixes for individual tasks rather than improve the agent system generally.

        The LLM should only see aggregate error-type statistics and agent
        attribution so it designs ARCHITECTURAL improvements (better prompting,
        stronger reasoning, pipeline topology changes) that lift performance
        across ALL future tasks, not just the ones that failed.
        """
        topology = getattr(parent_config, "topology", None)
        # Convert error breakdown to percentages so the LLM reasons in patterns
        error_breakdown = dict(getattr(analysis_report, "failure_breakdown", {}))
        total_failed = max(1, analysis_report.total_tasks - analysis_report.passed)
        error_pct = {
            err: f"{count}/{total_failed} failures ({count/total_failed:.0%})"
            for err, count in error_breakdown.items()
        }
        data = {
            "summary": (
                f"{analysis_report.passed}/{analysis_report.total_tasks} tasks passed "
                f"({analysis_report.pass_rate:.1%})"
            ),
            "error_type_distribution": error_pct,
            "agent_failure_attribution": getattr(analysis_report, "agent_failure_counts", {}),
            "current_topology": topology.to_dict() if topology else {},
            "note": (
                "These are AGGREGATE error patterns across all tasks. "
                "Do NOT reference specific task names or inputs. "
                "Design improvements that raise the pass rate for ANY coding task."
            ),
        }
        return json.dumps(data, indent=2)

    def _load_current_source(self) -> Dict[str, str]:
        """Load key source files for structural reference (truncated)."""
        priority = [
            "agent_coder.py", "agent_analyst.py", "pipeline.py",
            "config.py", "contracts.py", "llm_client.py",
        ]
        source = {}
        for fname in priority:
            fpath = self.gen_dir / fname
            if fpath.exists():
                try:
                    full = fpath.read_text(encoding="utf-8")
                    source[fname] = full[:500]
                except Exception:
                    pass
        logger.info(f"Loaded {len(source)} reference files for evolution context.")
        return source

    def _extract_code(self, response: str, fname: str) -> str:
        """
        Extract Python code from the LLM response.
        Handles:
          - Raw code (starts with # or import or \"\"\")
          - Fenced code blocks (```python ... ```)
          - BEGIN/END delimiters from old format
        """
        # Try BEGIN/END delimiter format
        m = re.search(
            r"=== BEGIN " + re.escape(fname) + r" ===\n(.*?)\n=== END " + re.escape(fname) + r" ===",
            response, re.DOTALL
        )
        if m:
            return m.group(1).strip()

        # Try fenced code block
        m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Raw code — strip any leading/trailing prose
        lines = response.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('"""') or stripped.startswith("'''") or
                    stripped.startswith("import ") or stripped.startswith("from ") or
                    stripped.startswith("# ")):
                start = i
                break
        return "\n".join(lines[start:]).strip()
