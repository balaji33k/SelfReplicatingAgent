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

# Core orchestration files the LLM MUST always generate — these are the minimum
# needed to run and evolve. Agent/pipeline files beyond this are decided by the LLM.
# The LLM declares its full file manifest in the planning phase (## FILE MANIFEST section).
CORE_FILES = [
    "main.py", "config.py", "task_manager.py",
    "analysis.py", "evolution_engine.py", "spawner.py",
]

# Delay between per-file LLM calls to stay inside TPM budget (seconds)
INTER_FILE_DELAY = 15


@dataclass
class NextGenerationDesign:
    """Holds the complete design for the next generation."""
    new_config: Config
    new_prompts_content: Dict[str, str] = field(default_factory=dict)
    improvement_log_message: str = ""
    evolution_artifacts: dict = field(default_factory=dict)  # full audit trail


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
        self,
        analysis_report: AnalysisReport,
        parent_config: Config,
        plan_callback=None,
    ) -> NextGenerationDesign:
        """
        plan_callback(event, payload) — optional hook so callers can react live:
          event="plan_ready"   payload={"plan": str, "files_total": int, "failure_summary": dict}
          event="file_done"    payload={"file": str, "index": int, "total": int}
          event="all_done"     payload={"files_generated": int}
        """
        gen_num = parent_config.generation_number
        next_gen = gen_num + 1
        logger.info(f"Designing Generation {next_gen} from failure evidence (one file at a time)...")

        failure_ctx = self._build_failure_context(analysis_report, parent_config)
        # Only load contracts.py — the shared interface spec agents must honour.
        # We deliberately do NOT load other parent files into the LLM context
        # to prevent indirect copying of parent logic.
        contracts_spec = self._load_contracts_spec()

        # Phase 1: Improvement plan (includes FILE MANIFEST — LLM declares what files it will generate)
        self._last_plan_sections = {}
        improvement_log = self._get_improvement_plan(failure_ctx, gen_num, next_gen)
        plan_sections = getattr(self, "_last_plan_sections", {})
        logger.info(f"Improvement plan: {improvement_log[:120]}...")
        logger.info(f"Topology decision: {plan_sections.get('topology_decision', '')[:120]}...")

        # Build dynamic file list: core files + any additional files the LLM declared
        llm_manifest = self._parse_file_manifest(plan_sections.get("file_manifest", ""))
        all_files = list(CORE_FILES)
        for f in llm_manifest:
            if f not in all_files and f.endswith(".py") and f not in INFRA_FILES:
                all_files.append(f)
        logger.info(f"File manifest: {all_files} ({len(all_files)} files)")

        # Notify caller as soon as the plan is ready (before any files are generated)
        if plan_callback:
            try:
                plan_callback("plan_ready", {
                    "plan": improvement_log,
                    "files_total": len(all_files),
                    "failure_summary": {
                        "pass_rate": analysis_report.pass_rate,
                        "passed": analysis_report.passed,
                        "total": analysis_report.total_tasks,
                        "error_breakdown": analysis_report.failure_breakdown,
                        "agent_attribution": analysis_report.agent_failure_counts,
                        "missing_packages": getattr(analysis_report, "missing_packages", []),
                    },
                })
            except Exception:
                pass

        # Phase 2: Generate each file — NO parent code shown, only the plan + stats
        files: Dict[str, str] = {}
        file_notes: Dict[str, str] = {}   # per-file generation notes for audit trail
        for i, fname in enumerate(all_files):
            logger.info(f"  Generating [{i+1}/{len(all_files)}]: {fname}")
            content = self._generate_one_file(
                fname=fname,
                improvement_log=improvement_log,
                failure_ctx=failure_ctx,
                contracts_spec=contracts_spec,
                gen_num=gen_num,
                next_gen=next_gen,
            )
            files[fname] = content
            file_notes[fname] = (
                f"Generated for Gen {next_gen}. "
                f"Role: {self._file_role(fname)}. "
                f"Addressed: {plan_sections.get('instruction_changes', '')[:100]}"
            )
            if plan_callback:
                try:
                    plan_callback("file_done", {
                        "file": fname,
                        "index": i + 1,
                        "total": len(all_files),
                    })
                except Exception:
                    pass
            if i < len(all_files) - 1:
                # No delay needed for local GPU model (Kaggle) — delay only for cloud API rate limits
                import os
                on_kaggle = bool(os.environ.get("KAGGLE_DATA_PROXY_TOKEN") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
                if not on_kaggle:
                    time.sleep(INTER_FILE_DELAY)

        missing = [f for f in CORE_FILES if f not in files]
        if missing:
            raise RuntimeError(f"Evolution incomplete — missing core files: {missing}")

        # Build full evolution artifact — saved to offspring dir by spawner
        import datetime as _dt
        evolution_artifacts = {
            "parent_generation": gen_num,
            "child_generation": next_gen,
            "timestamp": _dt.datetime.now().isoformat(),
            "failure_analysis": plan_sections.get("failure_analysis", ""),
            "topology_decision": plan_sections.get("topology_decision", ""),
            "instruction_changes": plan_sections.get("instruction_changes", ""),
            "expected_improvement": plan_sections.get("expected_improvement", ""),
            "full_improvement_plan": plan_sections.get("raw", improvement_log),
            "file_manifest": all_files,
            "files_generated": list(files.keys()),
            "file_notes": file_notes,
            "failure_stats": {
                "pass_rate": analysis_report.pass_rate,
                "passed": analysis_report.passed,
                "total": analysis_report.total_tasks,
                "error_breakdown": dict(getattr(analysis_report, "failure_breakdown", {})),
            },
        }

        logger.info(f"Generation {next_gen} design complete. {len(files)} files generated.")
        new_config = parent_config
        new_config.improvement_log = improvement_log
        return NextGenerationDesign(
            new_config=new_config,
            new_prompts_content=files,
            improvement_log_message=improvement_log,
            evolution_artifacts=evolution_artifacts,
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
      AssertionError       → reasoning/logic gap in code generation (add chain-of-thought)
      SyntaxError          → markdown fences not stripped, or LLM generating prose instead of code
      ModuleNotFoundError  → package not installed in sandbox; must auto-install before running code
      ImportError          → same as ModuleNotFoundError
      TimeoutError         → generated code has poor complexity; prompt must ask for complexity analysis
      RuntimeError         → edge cases unhandled; review/debug cycle may need enabling

Write a structured improvement plan with these EXACT sections:

## FAILURE ANALYSIS
What systemic weaknesses the error-type distribution reveals (2-3 sentences).

## TOPOLOGY DECISION
How many pipeline stages/agents you are designing for Generation {next_gen}, what each one is
responsible for, and WHY this topology addresses the observed failures.
Explain: why this number of agents, why this order, which stages can be skipped/looped.
You decide the topology — do not feel constrained to any prior agent names or count.

## FILE MANIFEST
List every .py file you will generate for Generation {next_gen}, one per line, format: filename.py
REQUIRED core files (always include): main.py, config.py, task_manager.py, analysis.py, evolution_engine.py, spawner.py
Additional files: list any pipeline/agent/utility files you design — use whatever names fit your topology.
Example (your actual files may differ):
  main.py
  config.py
  task_manager.py
  analysis.py
  evolution_engine.py
  spawner.py
  pipeline.py
  agent_solver.py
  agent_verifier.py

## INSTRUCTION CHANGES
For each pipeline stage, what specific instruction change you are making compared to a naive
approach, and why that change addresses a failure pattern from the statistics above.

## EXPECTED IMPROVEMENT
What pass-rate improvement you expect in Generation {next_gen} and why.

ARCHITECTURE CONSTRAINTS (must be preserved):
- llm_client.py is INFRASTRUCTURE — copied automatically, do NOT regenerate it
- The LLM client is already configured — use it as-is in your agents
- You decide the pipeline topology (number of agents, their roles, their order)
- You decide what frameworks or patterns to use (state machine, DAG, custom loop, etc.)
- Core files (main.py, config.py, task_manager.py, analysis.py, evolution_engine.py, spawner.py) are always required

MAS DESIGN PRINCIPLES (apply to every generation — do not remove or weaken):
1. LOOPS must have TWO exit conditions:
   a. Objective satisfaction: loop exits when a verifiable goal is met (e.g. tests pass,
      score improves, error type changes) — NOT when an agent subjectively says "looks good"
   b. Resource ceiling: a hard count limit prevents infinite loops when goal is unreachable
2. NO-PROGRESS DETECTION: if a loop iteration produces the same failure type as the
   previous iteration, exit immediately — the agent is stuck and more cycles waste tokens
3. PARALLEL BRANCHES (where LangGraph allows): independent sub-tasks should be designed
   to run concurrently and converge on the best result, not execute sequentially
4. SKIP UNSOLVABLE TASKS EARLY: before invoking any LLM agent, check whether the task
   is structurally solvable (has test cases, required packages available, non-empty
   problem statement) — skip immediately with a clear reason if not
5. SATISFACTION IS OBJECTIVE: "done" means tests pass, not "an agent says it's done"

GENERATIONAL ISOLATION — BIOLOGICAL MODEL (non-negotiable — every generation must implement):
The offspring generation follows the ectogenesis model — it is born outside the parent.

GESTATION (designing the next generation):
  - The evolution engine may run inside the current generation's environment while writing
    offspring code. This is fine — shared filesystem during gestation is expected.
  - BUT the LLM designing the next generation must receive ONLY failure analysis (DNA).
  - The LLM must NEVER see parent .py source files — no code snippets, no class definitions.
  - It designs from: pass rate, error breakdown, failure patterns, goals, design principles.
  - This forces true redesign from first principles, not copy-editing.

BIRTH (spawning the offspring):
  - The offspring must be written to a directory OUTSIDE the parent's directory tree.
  - Correct:   /generations/gen_N+1/   (sibling — same level as parent)
  - Wrong:     /generations/gen_N/generations/gen_N+1/   (nested inside parent)
  - The spawner must place every generation at the same depth, never nested.

POST-BIRTH (independence):
  - Once spawned, the offspring has NO dependency on its parent's filesystem.
  - The offspring must never import, read, or reference files from any previous generation.
  - The parent's directory may be deleted after birth — the offspring must not care.

UMBILICAL CORD — DNA PACKET:
  - At spawn time, the spawner writes dna.json into the offspring's directory.
  - dna.json contains: parent_pass_rate, error_breakdown, failure_summary, improvement_goals.
  - The offspring reads dna.json at startup to know what it must improve.
  - dna.json is the ONLY information channel between parent and child.

EVOLUTION ENGINE REQUIREMENT (critical — the plan you write must carry this forward):
  The evolution_engine.py you design for the next generation must:
  1. NEVER include parent source code in its generation prompts (DNA-only transfer)
  2. Include these biological isolation principles in its own improvement plan prompt
     so they propagate to every future generation without degradation
  3. Write dna.json into every offspring it spawns
  4. Place offspring as a sibling directory (same level), never nested inside the parent """

        response = self.llm_client.call(prompt)
        full_plan = response.strip()

        # Parse structured sections for audit trail
        self._last_plan_sections = self._parse_plan_sections(full_plan)

        return full_plan[:1500]  # truncated for inline use; full saved in artifacts

    def _parse_plan_sections(self, plan_text: str) -> dict:
        """
        Parse the structured improvement plan into named sections.
        Sections are ## HEADING format. Missing sections get empty string.
        """
        sections = {
            "failure_analysis": "",
            "topology_decision": "",
            "file_manifest": "",
            "instruction_changes": "",
            "expected_improvement": "",
            "raw": plan_text,
        }
        current_key = None
        current_lines = []
        key_map = {
            "FAILURE ANALYSIS":    "failure_analysis",
            "TOPOLOGY DECISION":   "topology_decision",
            "FILE MANIFEST":       "file_manifest",
            "INSTRUCTION CHANGES": "instruction_changes",
            "EXPECTED IMPROVEMENT":"expected_improvement",
        }
        for line in plan_text.splitlines():
            stripped = line.strip()
            matched = False
            for heading, key in key_map.items():
                if stripped.startswith("##") and heading in stripped.upper():
                    if current_key:
                        sections[current_key] = "\n".join(current_lines).strip()
                    current_key = key
                    current_lines = []
                    matched = True
                    break
            if not matched and current_key:
                current_lines.append(line)
        if current_key:
            sections[current_key] = "\n".join(current_lines).strip()
        return sections

    def _parse_file_manifest(self, manifest_text: str) -> List[str]:
        """
        Extract a list of .py filenames from the ## FILE MANIFEST section.
        Accepts any line containing 'filename.py' — strips bullets, whitespace.
        Returns only valid .py filenames that are not infra files.
        Falls back to an empty list if the section is missing/malformed
        (CORE_FILES will still be generated in that case).
        """
        files = []
        for line in manifest_text.splitlines():
            # Extract token that looks like a python filename
            import re
            m = re.search(r'\b([\w]+\.py)\b', line)
            if m:
                fname = m.group(1)
                if fname not in INFRA_FILES and fname not in files:
                    files.append(fname)
        if files:
            logger.info(f"[manifest] LLM declared {len(files)} files: {files}")
        else:
            logger.warning("[manifest] LLM did not provide a FILE MANIFEST — using CORE_FILES only")
        return files

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
- Use the LLMClient class (from llm_client import LLMClient) to make all LLM calls
- Every agent file must own its prompt and logic internally
- You choose the pipeline topology and framework — use whatever design best addresses failures
- No external frameworks are mandated — design from first principles

MAS DESIGN PRINCIPLES (non-negotiable — every generation must implement these):
- Every loop needs TWO exits: (a) objective satisfaction signal — e.g. tests pass,
  error type changed, score improved — not an agent's subjective opinion; (b) hard
  count ceiling to prevent infinite loops when the goal is unreachable
- No-progress detection: if a loop produces the same failure as the previous iteration,
  exit that loop immediately — repeated identical failures mean the agent is stuck
- Parallel branches: where sub-tasks are independent, design them to run concurrently;
  converge on the first result that satisfies the objective condition
- Skip unsolvable tasks before any LLM call: check for missing test cases, unavailable
  packages, or empty problem statements and return a clear skip result immediately
- Satisfaction is always objective and verifiable, never an agent's self-assessment

GENERATIONAL ISOLATION — BIOLOGICAL MODEL (must be preserved in every generated file):
{self._biological_isolation_block(fname)}

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
            # Core orchestration files — always generated
            "main.py":              "Entry point — loads tasks, runs pipeline, triggers evolution and spawn",
            "config.py":            "Configuration — LLMConfig, model defaults, generation metadata",
            "task_manager.py":      "Task loading from problem_pool.json",
            "analysis.py":          "Failure analysis — classifies errors, computes pass rate, attributes failures",
            "evolution_engine.py":  "Evolution — gets improvement plan + file manifest from LLM, generates each file",
            "spawner.py":           "Spawner — validates core files present, writes offspring dir, launches next gen",
            # Common pipeline/agent files (may or may not exist in this topology)
            "pipeline.py":          "Pipeline orchestrator — coordinates agent execution per the topology decision",
            "contracts.py":         "Data contracts — shared typed structures between pipeline stages",
            "agent_base.py":        "Base class for pipeline agents",
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
        missing_pkgs = getattr(analysis_report, "missing_packages", [])
        data = {
            "summary": (
                f"{analysis_report.passed}/{analysis_report.total_tasks} tasks passed "
                f"({analysis_report.pass_rate:.1%})"
            ),
            "error_type_distribution": error_pct,
            "agent_failure_attribution": getattr(analysis_report, "agent_failure_counts", {}),
            "current_topology": topology.to_dict() if topology else {},
            "missing_packages_detected": missing_pkgs,
            "sandbox_fix_required": (
                f"The sandbox failed to import: {missing_pkgs}. "
                "The next generation's sandbox_execution.py MUST auto-install missing packages "
                "using subprocess pip before executing generated code. "
                "Pattern: catch ImportError/ModuleNotFoundError, extract package name, "
                "run subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], ...), retry."
            ) if missing_pkgs else "No missing packages detected.",
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

    def _biological_isolation_block(self, fname: str) -> str:
        """
        Returns the appropriate biological isolation guidance for this file type.
        These principles MUST be included in every generation's prompts so they
        propagate forward without degradation (like genetic encoding).
        """
        if fname == "evolution_engine.py":
            return (
                "THIS FILE IS THE EVOLUTION ENGINE — it must enforce the ectogenesis model:\n"
                "1. DNA-ONLY TRANSFER: the LLM prompt you build must contain ONLY failure analysis,\n"
                "   improvement goals, design principles, and dna.json data. It must NEVER include\n"
                "   any .py source file, code snippet, or function definition from the parent gen.\n"
                "   Rationale: showing source code causes copy-editing, not redesign.\n"
                "2. PROPAGATE THESE PRINCIPLES: your improvement plan prompt must include the\n"
                "   biological isolation section so Gen N+2, N+3, ... all follow these rules.\n"
                "3. WRITE dna.json: call spawner with the DNA packet so each offspring reads it\n"
                "   at startup to understand what it must improve over the parent."
            )
        elif fname == "spawner.py":
            return (
                "THIS FILE IS THE SPAWNER — it must implement true ectogenesis birth:\n"
                "1. SIBLING BIRTH: the offspring directory must be at the SAME level as the parent.\n"
                "   Correct:  /generations/gen_N+1/  (sibling)\n"
                "   Wrong:    /generations/gen_N/generations/gen_N+1/  (nested inside parent)\n"
                "   Use: Path(__file__).resolve().parent.parent / f'gen_{next_gen_number}'\n"
                "2. WRITE dna.json INTO OFFSPRING: at spawn time, write a JSON file containing\n"
                "   parent_pass_rate, error_breakdown, failure_summary, improvement_goals.\n"
                "   This is the ONLY information channel between parent and child.\n"
                "3. NO PARENT CODE IN OFFSPRING: never copy parent .py logic files — only\n"
                "   infrastructure (llm_client.py, telemetry.py) and dna.json cross the boundary.\n"
                "4. POST-BIRTH INDEPENDENCE: the offspring must run without any reference to\n"
                "   parent directories. The parent dir may be deleted — offspring must not care."
            )
        elif fname == "main.py":
            return (
                "THIS FILE IS THE ENTRY POINT — it must read dna.json at startup:\n"
                "1. On startup, look for dna.json in the generation's own directory.\n"
                "2. If found, log the parent pass rate and improvement goals so there is a\n"
                "   clear audit trail of what this generation was designed to improve.\n"
                "3. The offspring never reads files from any previous generation directory.\n"
                "4. If dna.json is missing, log a warning but continue — Gen 1 has no parent."
            )
        else:
            return (
                "This file is part of an ectogenesis-model self-replicating system:\n"
                "- The parent's source code was NOT provided — design from first principles.\n"
                "- This generation was seeded from dna.json (failure analysis only), not parent code.\n"
                "- When this generation spawns the next, it must follow the same DNA-only rule:\n"
                "  no .py source files cross generation boundaries, only failure analysis data."
            )

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
