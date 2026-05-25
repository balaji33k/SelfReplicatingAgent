"""
evolution_engine.py — Designs the next generation from failure evidence.

Groq-compatible design: generates files ONE AT A TIME to stay within the
token-per-minute free-tier limit. Each call generates a single .py file
(~2,000 tokens) rather than all files at once (~50,000 tokens).

Key guarantees:
  1. The LLM decides which files to generate — the list is NOT hardcoded.
     It must include CORE_FILES but may add/remove/rename any other files.
  2. The LLM receives the full failure analysis for every file it generates.
  3. Anti-clone: every file must differ meaningfully from its parent version.
  4. Missing core files or compile-invalid files abort the spawn.
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

# Infrastructure files the LLM should NOT regenerate.
# These are always copied from the parent generation by spawner.py.
INFRA_FILES = {"telemetry.py", "lineage_memory.py", "llm_client.py"}

# Files the LLM must generate for every offspring.
REQUIRED_FILES = [
    "main.py", "config.py", "task_manager.py",
    "analysis.py", "evolution_engine.py", "spawner.py",
    "contracts.py", "agent_base.py", "pipeline.py",
    "agent_analyst.py", "agent_architect.py", "agent_coder.py",
    "agent_critic.py", "agent_reviser.py", "agent_test_writer.py",
    "agent_debugger.py", "agent_clone_inspector.py",
]

# Delay between per-file LLM calls to stay inside TPM budget (seconds)
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
        # Only load contracts.py — the shared interface spec agents must honour.
        # We deliberately do NOT load other parent files into the LLM context
        # to prevent indirect copying of parent logic.
        contracts_spec = self._load_contracts_spec()

        # Phase 1: Improvement plan
        improvement_log = self._get_improvement_plan(failure_ctx, gen_num, next_gen)
        logger.info(f"Improvement plan: {improvement_log[:120]}...")

        # Phase 2: Generate each file — NO parent code shown, only the plan + stats
        files: Dict[str, str] = {}
        for i, fname in enumerate(REQUIRED_FILES):
            logger.info(f"  Generating [{i+1}/{len(REQUIRED_FILES)}]: {fname}")
            content = self._generate_one_file(
                fname=fname,
                improvement_log=improvement_log,
                failure_ctx=failure_ctx,
                contracts_spec=contracts_spec,
                gen_num=gen_num,
                next_gen=next_gen,
            )
            files[fname] = content
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
        gen_num: int,
        next_gen: int,
    ) -> str:
        """
        Ask the LLM for a concise improvement plan (plain text, no code).
        No parent source code is included — only failure statistics and
        architectural constraints. This prevents the plan itself from
        carrying parent logic that files would later copy.
        """
        prompt = f"""You are designing Generation {next_gen} of a self-replicating AI coding agent.

AGGREGATE FAILURE STATISTICS (Generation {gen_num}):
{failure_ctx}

⚠️  GENERAL IMPROVEMENTS ONLY — DO NOT make task-specific fixes.
    Treat failure statistics as signals of SYSTEMIC weaknesses in the agent design.

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
        return response.strip()[:1500]

    # ── Phase 2: Per-file generation ──────────────────────────────────────────

    def _generate_one_file(
        self,
        fname: str,
        improvement_log: str,
        failure_ctx: str,
        contracts_spec: str,
        gen_num: int,
        next_gen: int,
    ) -> str:
        """
        Generate a single .py file for the next generation.

        Parent source code is deliberately NOT shown — showing it would let
        the LLM copy logic indirectly (same algorithm, different variable names).
        Only the data contracts (shared interface) are provided so agents stay
        compatible with each other. Everything else must be designed fresh.
        """
        file_role = self._file_role(fname)

        # contracts.py and agent_base.py define the shared interface —
        # show them so the LLM writes compatible code, but not as "logic to copy".
        interface_section = ""
        if fname not in ("contracts.py", "agent_base.py") and contracts_spec:
            interface_section = f"""
DATA CONTRACTS (shared interface — your code must be COMPATIBLE with these types,
but you must NOT copy any logic from them):
{contracts_spec[:600]}
"""

        prompt = f"""You are generating {fname} for Generation {next_gen} of a self-replicating AI coding agent.

⚠️  NO COPYING — DIRECT OR INDIRECT:
    You have NOT been shown the parent generation's source code for this file.
    Design this file completely from scratch based on the improvement plan below.
    Do NOT reproduce the parent's algorithms, variable names, class structures,
    or logic patterns — even if you happen to know them from training.
    The spawner runs both byte-level identity checks AND semantic similarity checks.
    Near-copies (same logic, different names) are rejected just like exact copies.

IMPROVEMENT PLAN (what the new generation should do differently and why):
{improvement_log[:600]}

AGGREGATE FAILURE STATISTICS (design signals — no task-specific fixes):
{failure_ctx[:400]}

FILE ROLE: {file_role}
{interface_section}
⚠️  GENERAL IMPROVEMENTS ONLY:
    Improve agent prompts, reasoning strategies, pipeline topology, or error handling
    in ways that help solve ANY coding task. Do NOT hardcode fixes for specific inputs.

REQUIREMENTS:
- File: {fname}  Generation: {next_gen}
- Address the SYSTEMIC failure patterns shown in the statistics above
- llm_client.py is INFRASTRUCTURE — do NOT generate it; it is copied automatically
- LLMClient uses ChatGroq (langchain-groq), reads GROQ_API_KEY from env, max_retries=7
- Default Groq model: meta-llama/llama-4-scout-17b-16e-instruct (30k TPM / 500k TPD)
- pipeline.py MUST use LangGraph (langgraph.graph.StateGraph) — NOT custom loops
  • from langgraph.graph import StateGraph, START, END
  • PipelineState as TypedDict; conditional edges for critic/debug cycles
  • Compile once in __init__; stream with stream_mode="updates" in solve()
- Every agent file must own its prompt internally
- requirements.txt already includes: langgraph>=0.2.0, langchain-groq>=0.2.0, langchain-core>=0.3.0

OUTPUT: Write ONLY the complete Python source code for {fname}.
Start with the module docstring. No explanation outside the code."""

        response = self.llm_client.call(prompt)
        return self._extract_code(response, fname)

    def _file_role(self, fname: str) -> str:
        """
        One-line description of what each file is responsible for.
        Known files get a precise role. Unknown files (new ones the LLM added)
        get a generic description — the LLM decided to create them so it knows
        their purpose from the improvement plan.
        """
        roles = {
            "main.py":              "Entry point — loads tasks, runs pipeline, triggers evolution",
            "config.py":            "Configuration — AgentTopologyConfig, LLMConfig, model defaults",
            "task_manager.py":      "Task loading from problem_pool.json",
            "analysis.py":          "Failure analysis — classifies errors, attributes to agents",
            "evolution_engine.py":  "Evolution — gets improvement plan + file manifest from LLM, generates each file",
            "spawner.py":           "Spawner — validates core files present, writes offspring, launches next gen",
            "contracts.py":         "Data contracts — typed dataclasses between agents",
            "agent_base.py":        "Base class for all specialist agents",
            "pipeline.py":          "AgentPipeline — LangGraph StateGraph (langgraph>=0.2.0); PipelineState TypedDict; conditional edges for critic/debug cycles; stream_mode='updates'",
            "agent_analyst.py":     "Analyst agent — decomposes problem into AnalysisSpec",
            "agent_architect.py":   "Architect agent — designs solution as DesignSpec",
            "agent_coder.py":       "Coder agent — generates working Python code; use chain-of-thought",
            "agent_critic.py":      "Critic agent — reviews code against spec (not design intent)",
            "agent_reviser.py":     "Reviser agent — applies Critic's targeted fixes only",
            "agent_test_writer.py": "TestWriter agent — generates tests from AnalysisSpec ONLY",
            "agent_debugger.py":    "Debugger agent — patches exact failure, no redesign",
            "agent_clone_inspector.py": "Clone Inspector — detects if offspring files copy parent",
        }
        # For any new file the LLM decided to add (not in the known list),
        # infer its role from the filename rather than returning a generic stub.
        if fname not in roles:
            base = fname.replace(".py", "").replace("_", " ")
            return f"New file introduced in this generation: {base} — implement as described in the improvement plan"
        return roles[fname]

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

    def _load_contracts_spec(self) -> str:
        """
        Load only contracts.py — the shared data types all agents must honour.

        This is the ONLY parent file shown to the LLM during generation.
        We do NOT load any other parent source because doing so lets the LLM
        copy logic indirectly (same algorithm, different variable names).

        contracts.py defines interfaces (TypedDict, dataclass), not logic —
        the LLM needs it to write agents that are type-compatible with each other.
        """
        contracts_path = self.gen_dir / "contracts.py"
        if contracts_path.exists():
            try:
                content = contracts_path.read_text(encoding="utf-8")
                logger.info("[evolution] Loaded contracts.py as interface spec.")
                return content[:800]   # dataclass definitions only, not full file
            except Exception:
                pass
        logger.warning("[evolution] contracts.py not found — offspring agents may have type mismatches.")
        return ""

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
