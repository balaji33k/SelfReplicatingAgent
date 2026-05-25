# Self-Replicating AI Coding Agent — Complete System Document

> **Purpose:** This document describes every component, design decision, data flow, guardrail,
> and operational detail of the self-replicating multi-agent AI coding system.
> It is the single source of truth for understanding how the system works.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Deployment & Infrastructure](#3-deployment--infrastructure)
4. [The Multi-Agent Pipeline](#4-the-multi-agent-pipeline)
5. [Agent Roles & Data Contracts](#5-agent-roles--data-contracts)
6. [LangGraph StateGraph Internals](#6-langgraph-stategraph-internals)
7. [LLM Client & Token Tracking](#7-llm-client--token-tracking)
8. [Evolution Engine — How Offspring Are Designed](#8-evolution-engine--how-offspring-are-designed)
9. [Anti-Clone Guardrails](#9-anti-clone-guardrails)
10. [Spawner — Creating the Next Generation](#10-spawner--creating-the-next-generation)
11. [Failure Analysis](#11-failure-analysis)
12. [Configuration & Topology](#12-configuration--topology)
13. [Problem Pool](#13-problem-pool)
14. [Dashboard — MISSION_CONTROL.html](#14-dashboard--mission_controlhtml)
15. [Data Files Written at Runtime](#15-data-files-written-at-runtime)
16. [Key Design Decisions & Why](#16-key-design-decisions--why)
17. [What Each Generation Can Change](#17-what-each-generation-can-change)
18. [File Reference](#18-file-reference)

---

## 1. What This System Does

This is a **self-replicating AI coding agent**. It runs a benchmark of coding tasks,
measures its own performance, identifies the root causes of failures, and then uses
an LLM to design an improved version of itself — which it writes to disk and launches
as the next generation. This process repeats indefinitely.

**Core loop:**
```
Gen N solves benchmark tasks
    → analyses failures (error types, which agents failed)
    → LLM designs Gen N+1 (17 .py files, no parent code shown)
    → spawner validates (no direct copy, no indirect copy)
    → Gen N+1 is written to disk and launched
    → Gen N exits
    → Gen N+1 repeats the loop
```

**Goal:** Each generation should have a higher benchmark pass rate than its parent,
by making architectural improvements to agent prompts, reasoning strategies, and
pipeline topology — not by memorising specific tasks.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HuggingFace Space                           │
│                                                                     │
│   serve.py                                                          │
│   ├── HTTP Server (port 7860) ──→ MISSION_CONTROL.html (dashboard) │
│   └── Evolution Thread                                              │
│         └── python gen_1/main.py                                   │
│               ├── TaskLoader   (loads 20 benchmark tasks)           │
│               ├── AgentPipeline (LangGraph, 8 agents)              │
│               │     └── solves each task                           │
│               ├── FailureAnalyzer (classifies errors)              │
│               ├── EvolutionEngine (designs Gen 2 via LLM)          │
│               └── Spawner (writes gen_2/, launches it)             │
│                     └── python gen_2/main.py  (independent proc)  │
│                           └── ... (same loop, Gen 2 → Gen 3 ...)  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Every generation is a completely independent process in its own
directory. No generation modifies another generation's files.

---

## 3. Deployment & Infrastructure

### Platform
- **HuggingFace Spaces** — Docker SDK, free tier
- Port: **7860** (required by HF)
- Filesystem: ephemeral (wiped on restart, except for HF Dataset uploads)

### Git Repository
- Branch: `fresh-main` (orphan branch, pushed as `main` to HF)
- Push command: `git push hf fresh-main:main --force`
- **Never push to `main` locally** — only to HF remote

### Secrets
- `GROQ_API_KEY` — stored as HF Space secret, never hardcoded
- Never commit API keys to git

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user . .
RUN mkdir -p data generations/gen_1/results generations/gen_1/sandbox
ENV PORT=7860
CMD ["python", "serve.py"]
```

### serve.py — Entry Point
Two threads run simultaneously:
1. **HTTP server** — serves `MISSION_CONTROL.html`, `data/*.json`, generation source files
2. **Evolution thread** — runs `gen_1/main.py`, repeats every 5 minutes after completion

On each evolution start, `serve.py` clears all volatile runtime files:
- `data/heartbeat.json`, `data/evolution_log.json`, `data/token_usage.json`
- `data/benchmark_matrix.json`, `data/gen_*_progress.json`
- `generations/gen_1/results/*.json`

### requirements.txt
```
huggingface_hub>=0.23.0
langgraph>=0.2.0
langchain-groq>=0.2.0
langchain-core>=0.3.0
typing-extensions>=4.0.0
```

---

## 4. The Multi-Agent Pipeline

### Overview
Each task passes through a pipeline of specialist agents, each with a single
cognitive responsibility. The pipeline is built as a **LangGraph StateGraph** —
a directed graph with conditional edges for revision and debug cycles.

### Pipeline Flow
```
START
  → Analyst       (decomposes problem)
  → Architect     (designs algorithm)
  → [TestWriter?] (generates extra tests — optional)
  → Coder         (writes Python code)
  → [Critic?]     (reviews code)
    ↕ cycle (up to max_revise_cycles)
  → [Reviser?]    (fixes Critic's findings)
  → Executor      (runs code in sandbox — NO LLM call)
  → [Debugger?]   (patches failures — optional)
    ↕ cycle (up to max_debug_cycles)
  → END
```

### Topology Configuration
`AgentTopologyConfig` in `config.py` controls which agents are active:

| Flag | Default (Gen 1) | Effect |
|---|---|---|
| `enable_critic` | `False` | Adds Critic → Reviser review cycle |
| `enable_reviser` | `False` | Required alongside enable_critic |
| `enable_test_writer` | `False` | Generates extra test cases from AnalysisSpec |
| `enable_debugger` | `False` | Patches code after sandbox failures |
| `max_revise_cycles` | `2` | Max times Critic→Reviser loop runs per task |
| `max_debug_cycles` | `1` | Max times Debugger runs per task |
| `execution_timeout` | `10` | Seconds per sandbox run |

**Gen 1 rationale:** Start minimal (Analyst→Architect→Coder→Executor only) to
establish a working baseline. Subsequent generations enable optional agents based
on what failure evidence shows.

### Information Isolation
Each agent sees only what it needs — never more:

| Agent | Receives | Does NOT see |
|---|---|---|
| Analyst | `TaskContract` (raw problem) | Code, tests |
| Architect | `AnalysisSpec` | Raw problem, test cases |
| TestWriter | `AnalysisSpec` ONLY | Code, design |
| Coder | `DesignSpec` + `AnalysisSpec` | Raw problem, test cases |
| Critic | `CodeArtifact` + `AnalysisSpec` | DesignSpec (avoids design bias) |
| Reviser | `CodeArtifact` + `CritiqueReport` | Original problem |
| Executor | `CodeArtifact` + test cases | LLM not called |
| Debugger | `CodeArtifact` + `ExecutionContract` | Original problem |

---

## 5. Agent Roles & Data Contracts

### Data Contracts (`contracts.py`)
All data passed between agents is typed. No agent ever receives raw strings
from the orchestrator — only typed dataclass instances.

```python
TaskContract      → input to pipeline (problem + test cases)
AnalysisSpec      → Analyst output (structured problem decomposition)
DesignSpec        → Architect output (algorithm design)
TestSuite         → TestWriter output (extra test cases)
CodeArtifact      → Coder output (Python code + function name)
CritiqueReport    → Critic output (passed: bool, issues list)
ExecutionContract → Executor output (success, stdout, stderr, results)
PipelineResult    → Final output of solve() (aggregates everything)
```

### Agent Descriptions

**AnalystAgent** (`agent_analyst.py`)
- Input: `TaskContract`
- Output: `AnalysisSpec`
- Job: Decompose the problem into structured fields — input types, output type,
  constraints, edge cases, success criteria. Never writes code.

**ArchitectAgent** (`agent_architect.py`)
- Input: `AnalysisSpec`
- Output: `DesignSpec`
- Job: Choose an algorithm (with rationale), define data structures, write
  pseudocode steps, estimate time/space complexity, plan edge case handling.

**TestWriterAgent** (`agent_test_writer.py`)
- Input: `AnalysisSpec` ONLY (never sees code — prevents circular reasoning)
- Output: `TestSuite` (extra test cases)
- Job: Generate additional test cases for edge cases not covered in the original pool.

**CoderAgent** (`agent_coder.py`)
- Input: `DesignSpec` + `AnalysisSpec`
- Output: `CodeArtifact`
- Job: Implement the algorithm as runnable Python. Uses chain-of-thought prompting.
  Strips markdown fences from output. Returns clean Python with no prose.

**CriticAgent** (`agent_critic.py`)
- Input: `CodeArtifact` + `AnalysisSpec` (NOT DesignSpec — avoids design bias)
- Output: `CritiqueReport`
- Job: Review code for correctness against the problem spec. Returns
  `passed=True` if code looks correct, or a list of specific issues.

**ReviserAgent** (`agent_reviser.py`)
- Input: `CodeArtifact` + `CritiqueReport`
- Output: `CodeArtifact` (revised version)
- Job: Apply only the Critic's specific fixes. No redesign. Targeted patches only.

**Executor** (built into `pipeline.py` — no LLM call)
- Input: `CodeArtifact` + all test cases
- Output: `ExecutionContract`
- Job: Run code in a subprocess sandbox with a timeout. Parse stdout as JSON
  test results. No LLM involved — purely deterministic.

**DebuggerAgent** (`agent_debugger.py`)
- Input: `CodeArtifact` + `ExecutionContract` (actual error output)
- Output: `CodeArtifact` (patched version)
- Job: Read the actual error/failure and patch the exact problem. No redesign.

**CloneInspectorAgent** (`agent_clone_inspector.py`)
- Input: parent files + child files dict
- Output: originality report (verdict, score, near-clone list)
- Job: Semantic check — is this offspring genuinely different from its parent,
  or is it a reworded copy? Called by Spawner before writing offspring to disk.

---

## 6. LangGraph StateGraph Internals

### Why LangGraph
Before LangGraph, the pipeline used a manual loop with `while` loops for
critic/debug cycles. LangGraph replaces this with a compiled graph that:
- Makes topology explicit and debuggable
- Handles cyclic edges natively (critic↔reviser, debugger↔executor)
- Supports `stream_mode="updates"` — fires a callback after each node, enabling
  live dashboard phase updates without polling

### PipelineState
A single `TypedDict` flows through all nodes. Each node returns only the fields
it changed — LangGraph merges them:

```python
class PipelineState(TypedDict, total=False):
    task:           TaskContract
    analysis:       Optional[AnalysisSpec]
    design:         Optional[DesignSpec]
    extra_tests:    Optional[TestSuite]
    artifact:       Optional[CodeArtifact]
    critique:       Optional[CritiqueReport]
    execution:      Optional[ExecutionContract]
    revise_count:   int
    debug_count:    int
    agents_used:    List[str]
    phase_results:  Dict[str, Optional[bool]]
    agent_failures: Dict[str, str]
```

### Node Factories
Each agent node is created by a factory function that closes over the agent instance:
```python
def _make_coder_node(agent):
    def node(state: PipelineState) -> dict:
        return {"artifact": agent.run(state["design"], state["analysis"]),
                "agents_used": state.get("agents_used", []) + ["coder"]}
    return node
```

### Conditional Edge Routers
Replace `while` loops:
```python
def _make_critic_router(max_cycles):
    def router(state) -> str:
        if critique and not critique.passed and count < max_cycles:
            return "reviser"   # ← cycle back
        return "executor"      # ← proceed
    return router

def _make_executor_router(max_cycles, has_debugger):
    def router(state) -> str:
        if has_debugger and not execution.success and count < max_cycles:
            return "debugger"  # ← cycle back
        return END             # ← done
    return router
```

### Graph Compilation
Built once at `AgentPipeline.__init__`, driven entirely by `AgentTopologyConfig`:
```python
g = StateGraph(PipelineState)
g.add_node("analyst", _make_analyst_node(agents["analyst"]))
# ... all nodes
if cfg.enable_critic:
    g.add_edge("reviser", "critic")         # cycle
    g.add_conditional_edges("critic", router, {...})
if cfg.enable_debugger:
    g.add_edge("debugger", "executor")      # cycle
    g.add_conditional_edges("executor", router, {...})
g.add_edge(START, "analyst")
return g.compile()
```

### Live Phase Callbacks
`solve()` streams node updates to the dashboard:
```python
for chunk in self._graph.stream(initial_state, stream_mode="updates"):
    for node_name, node_update in chunk.items():
        phase = _PHASE_MAP[node_name]   # e.g. "coder" → "writing_code"
        phase_callback(phase, node_update.get("phase_results", {}))
```

---

## 7. LLM Client & Token Tracking

### Provider
- **Groq** via `langchain-groq` (`ChatGroq` class)
- Model: `meta-llama/llama-4-scout-17b-16e-instruct`
- Limits: **500,000 tokens/day**, **30,000 tokens/minute** (Groq free tier)
- `max_retries=7` — LangChain handles exponential backoff automatically

### Interface
`LLMClient` exposes two methods used by all agents:
```python
def call(self, prompt: str, agent_name: str = "") -> str
def chat_completion(self, messages: List[Dict], agent_name: str = "") -> str
```
`agent_name` is passed by `agent_base._call_llm()` as `self.agent_id` so each
agent's token spend is tracked separately.

### Token Tracking
After every LLM call, `_record_usage()` extracts token counts:
```python
# Primary: langchain-core standard
meta = response.usage_metadata   # {"total_tokens": N}
# Fallback: Groq-specific
rmeta = response.response_metadata["token_usage"]["total_tokens"]
```

Counts are accumulated per agent in a thread-safe dict, then written to
`data/token_usage.json` after every call:
```json
{
  "model": "meta-llama/llama-4-scout-17b-16e-instruct",
  "provider": "groq",
  "date": "2026-05-25",
  "daily_limit_tokens": 500000,
  "tpm_limit": 30000,
  "total_tokens_used": 45200,
  "tokens_remaining": 454800,
  "pct_used": 0.0904,
  "per_agent": {
    "analyst":   {"tokens": 2340, "calls": 5},
    "architect": {"tokens": 3120, "calls": 5},
    "coder":     {"tokens": 12900, "calls": 5}
  }
}
```
Counter resets automatically at midnight (daily limit is per UTC day).

---

## 8. Evolution Engine — How Offspring Are Designed

### Two-Phase Process

**Phase 1 — Improvement Plan** (one LLM call, ~500 tokens)
The LLM receives:
- Aggregate failure statistics (error type percentages + agent attribution)
- Architecture constraints (LangGraph, ChatGroq, model name)
- Error-type → root-cause mapping to guide reasoning

The LLM does NOT receive:
- Any parent source code (prevents indirect copying)
- Specific task names, inputs, or expected outputs (prevents task-specific fixes)

Output: a plain-text improvement plan describing what to change and why.

**Phase 2 — Per-File Generation** (one LLM call per file, ~2,000 tokens each)
For each of the 17 required files:
- The LLM receives: improvement plan, failure statistics, file role description
- The LLM receives `contracts.py` content ONLY — as an interface spec to maintain compatibility
- The LLM does NOT receive: the parent's implementation of that file
- 15-second delay between calls to stay within 30,000 TPM limit

### Why No Parent Code Is Shown
Showing parent code (even with "do not copy" instructions) leads to indirect copying —
the LLM reproduces the same algorithm with different variable names. Instead:
- Phase 1 prompt: no source code at all
- Phase 2 prompt: only `contracts.py` (data types) — purely the interface, not logic

### Failure Context (Task-Agnostic)
The failure context passed to the LLM contains only aggregate statistics:
```json
{
  "summary": "5/20 tasks passed (25%)",
  "error_type_distribution": {
    "AssertionError": "10/15 failures (67%)",
    "TimeoutError": "3/15 failures (20%)",
    "ImportError": "2/15 failures (13%)"
  },
  "agent_failure_attribution": {"coder": 12, "analyst": 2},
  "current_topology": {"enable_critic": false, ...},
  "note": "AGGREGATE patterns only. Do NOT fix specific tasks."
}
```

Task names, specific inputs, and error messages are intentionally excluded.

### Inter-File Delay
```
30,000 TPM limit / ~2,500 tokens per call = 12 calls/min max
15-second gap between calls → 4 calls/min → safely within budget
```

---

## 9. Anti-Clone Guardrails

Three layers prevent offspring from copying parent code:

### Layer 1 — LLM Prompt (upstream prevention)
Every per-file generation prompt states:
> *"You have NOT been shown the parent generation's source code.
> Design this file completely from scratch.
> Do NOT reproduce the parent's algorithms, variable names, class structures,
> or logic patterns — even if you happen to know them from training.
> The spawner runs both byte-level identity checks AND semantic similarity checks.
> Near-copies (same logic, different names) are rejected just like exact copies."*

### Layer 2 — Spawner Direct + Indirect Copy Check
Applied to every generated `.py` file using `difflib.SequenceMatcher`:

| Detection | Threshold | Action |
|---|---|---|
| Direct copy | ratio = 1.0 (byte-identical) | `RuntimeError` — spawn aborted |
| Indirect copy | ratio > 0.85 (same logic, renamed) | `RuntimeError` — spawn aborted |

`contracts.py` and `agent_base.py` are exempt — these define interfaces that
legitimately change little between generations.

### Layer 3 — CloneInspectorAgent (semantic LLM check)
After byte/difflib validation, `agent_clone_inspector.py` does a semantic check:
- Compares parent vs child code using `difflib` for structural diff
- Asks LLM: *"Is this a genuine improvement or a reworded copy?"*
- `REJECT` verdict → spawn aborted with full explanation
- Near-clone warnings → logged but do not abort (byte-level is the hard gate)

### What Happens on Rejection
If any layer rejects the offspring, `RuntimeError` is raised before `gen_N+1/`
is created. The directory is never written. The current generation logs the failure
and the serve.py loop restarts evolution from Gen 1 after a 5-minute wait.

---

## 10. Spawner — Creating the Next Generation

### What the Spawner Does
1. **Validate** — check all 17 required files are present and pass copy checks
2. **Compile check** — `ast.parse()` every generated file; reject on `SyntaxError`
3. **Semantic check** — run `CloneInspectorAgent`
4. **Write** — create `generations/gen_N+1/` and write all LLM-generated files
5. **Copy infrastructure** — copy `llm_client.py`, `telemetry.py`, `lineage_memory.py`,
   `persist.py` from parent (these are never regenerated by the LLM)
6. **Copy problem pool** — copy `problem_pool.json` into new generation directory
7. **Write manifest** — record generation number, parent pass rate, parent error types,
   improvement log, timestamp, files generated
8. **Record evolution** — call `evolution_tracker.record_spawn()` to update
   `data/evolution_log.json` for the dashboard
9. **Launch** — `subprocess.Popen([sys.executable, "main.py"])` in `gen_N+1/` directory

### Infrastructure vs Logic Files

| Type | Files | Who generates |
|---|---|---|
| Logic (LLM-generated) | All 17 `.py` files | LLM (must differ from parent) |
| Infrastructure (copied) | `llm_client.py`, `telemetry.py`, `lineage_memory.py`, `persist.py` | Copied from parent |
| Data (copied) | `problem_pool.json` | Copied from project root or parent |

### Why llm_client.py Is Infrastructure
`llm_client.py` handles API connectivity (ChatGroq, retries, token tracking).
If the LLM regenerates it, it might revert to urllib, lose retry logic, or break
the token tracking. Keeping it as infrastructure ensures every generation gets
a working LLM connection without reimplementing it.

---

## 11. Failure Analysis

`FailureAnalyzer` in `analysis.py` processes all task results and produces
an `AnalysisReport` with:

### Error Type Classification
| Error Type | Detection |
|---|---|
| `TimeoutError` | `subprocess.TimeoutExpired` or stderr contains "Timeout" |
| `SyntaxError` | stderr contains "SyntaxError" |
| `ImportError` | stderr contains "ImportError" or "ModuleNotFoundError" |
| `AssertionError` | Code ran but test cases failed (`passed=False`) |
| `RuntimeError` | Any other non-zero exit code |
| `SystemError` | Pipeline-level exception (agent crashed) |

### Agent Attribution
Which agent caused which failures:
- `agent_failures` dict in `PipelineResult` — set when an agent raises an exception
- Used to show per-agent failure counts in the evolution engine context

### Output to Evolution Engine
```python
AnalysisReport:
    pass_rate: float
    passed: int
    total_tasks: int
    failure_breakdown: Dict[str, int]   # error_type → count
    agent_failure_counts: Dict[str, int]
    sample_failures: List[dict]         # NOT passed to LLM — task-specific
```
Only aggregate statistics go into the LLM prompts — never specific task details.

---

## 12. Configuration & Topology

### Config Hierarchy
```
Config
├── LLMConfig
│   ├── model_name: "meta-llama/llama-4-scout-17b-16e-instruct"
│   ├── temperature: 0.7
│   └── timeout_seconds: 300
├── SandboxConfig
│   ├── timeout_seconds: 60
│   └── memory_limit_mb: 512
└── AgentTopologyConfig
    ├── enable_critic: bool
    ├── enable_reviser: bool
    ├── enable_test_writer: bool
    ├── enable_debugger: bool
    ├── max_revise_cycles: int
    ├── max_debug_cycles: int
    └── execution_timeout: int
```

### How Topology Evolves
The LLM generates a new `config.py` each generation. It may enable additional
agents (critic, reviser, debugger, test_writer) if the failure evidence suggests
they would help. The pipeline.py reads topology at startup and builds the
LangGraph graph accordingly — no hardcoded topology anywhere.

### Manifest
Each generation writes `manifest.json` to its directory:
```json
{
  "GENERATION": 2,
  "PARENT_PASS_RATE": 0.25,
  "PARENT_ERRORS": {"AssertionError": 10, "TimeoutError": 3},
  "IMPROVEMENT_LOG": "Enabled chain-of-thought in Coder prompt...",
  "TIMESTAMP": "2026-05-25T10:30:00",
  "FILES_GENERATED_BY_LLM": ["main.py", "config.py", ...]
}
```

---

## 13. Problem Pool

### Location
`generations/gen_1/problem_pool.json` — copied to every offspring generation.

### Contents
**20 tasks total:**
- 10 SWE-bench tasks (xarray repository, ordered easiest→hardest)
- 10 LiveCodeBench tasks (LCB-001 through LCB-010, algorithmic problems)

### Task Structure
```json
{
  "task_id": "LCB-001",
  "description": "Given an array of integers...",
  "problem_statement": "Full problem text...",
  "test_cases": [
    {"input": [1, 2, 3], "expected": 6},
    {"input": [], "expected": 0}
  ],
  "signature": "def solve(nums: List[int]) -> int",
  "metadata": {"difficulty": "Easy", "category": "array"}
}
```

### Why 20 Tasks
- Reduced from 30 to stay within Groq's daily token budget (500k TPD)
- 20 tasks × ~8 LLM calls per task × ~2,000 tokens = ~320k tokens
- Leaves ~180k tokens for evolution (17 file generations × ~2,500 tokens = ~42.5k)

---

## 14. Dashboard — MISSION_CONTROL.html

### How It Works
A single static HTML file with vanilla JavaScript. No framework, no build step.
Polls three JSON files every 3 seconds:
- `data/heartbeat.json` — live status (gen number, progress, current phase)
- `data/evolution_log.json` — history of all generation→generation improvements
- `data/token_usage.json` — model name, daily budget, per-agent token counts

### Cards & Sections

**Generation Timeline** — clickable circles for each generation, shows pass rate

**Gen Detail** — for selected generation: stat grid, task list with pass/fail/error type, collapse/expand

**🤖 LLM Model & Daily Token Budget**
- Model name (full path)
- Provider badge + TPM/TPD limits
- Daily usage progress bar (green → amber → red)
- Per-agent breakdown: name | tokens used | proportional bar | call count

**🧬 Evolution Improvements**
- One card per generation transition (Gen N → Gen N+1)
- Left border: cyan=pending, green=improved, red=regressed
- Shows: pass rate delta chip, error type tags, topology change tags, full LLM improvement log

**Hero Progress Card**
- Active generation number + status badge (Running/Analysing/Spawning/Done)
- Circular progress indicator (% of tasks done)
- Main progress bar with pass/fail/pending segmentation
- Live task phase progress bar (Think→Code→Review→Compile→Debug steps)
- Pipeline topology diagram (nodes coloured by active/inactive)
- Live ticker showing current task ID
- Mini stats: passed, failed, pass rate, generation number

**Task Summary / Code Generated tabs** (shown while Running/Analysing)
- Summary: current task description, phase checks (compile ✓/✗, unit tests ✓/✗)
- Code: generated code for last completed task, with stderr if failed

### Live Updates
- Phase callbacks from `AgentPipeline.solve()` → `_write_heartbeat()` in `main.py`
- Dashboard reads `heartbeat.json` every 3 seconds → re-renders all cards
- Token budget card reads `token_usage.json` on same 3-second tick

---

## 15. Data Files Written at Runtime

| File | Written by | Contents |
|---|---|---|
| `data/heartbeat.json` | `main.py` after every phase | Status, progress, active task, phase, topology |
| `data/evolution_log.json` | `evolution_tracker.py` | Array of gen-to-gen improvement records |
| `data/token_usage.json` | `llm_client.py` after every LLM call | Model, budget, per-agent token counts |
| `data/benchmark_matrix.json` | `evolution_tracker.py` | Cross-generation task pass/fail matrix |
| `data/gen_N_progress.json` | `main.py` after every task | Task list, per-task status, descriptions |
| `generations/gen_N/results/*.json` | `main.py` after every task | Full PipelineResult for each task |
| `generations/gen_N/results/benchmark_summary.json` | `main.py` | AnalysisReport for the generation |
| `generations/gen_N/manifest.json` | `spawner.py` | Generation metadata |
| `generations/gen_N/generation.log` | Python `logging` | Full execution log |
| `generations/gen_N/child_generation.log` | `spawner.py` | Stdout of the child process |

### What Gets Cleared on Restart
`serve.py` clears all volatile data files before each evolution run so the
dashboard starts fresh with no stale data from a previous run.

---

## 16. Key Design Decisions & Why

### No Parent Code in Generation Prompts
**Problem:** Showing parent code with "don't copy" instructions still leads to
indirect copying — the LLM reproduces the same algorithm with different names.
**Decision:** Only `contracts.py` (data types) is shown as an interface spec.
All logic files are designed from scratch based on failure statistics and an
improvement plan.

### General Improvements Only
**Problem:** If the LLM sees specific task failures (e.g., "xarray `.where()` bug"),
it will hardcode fixes for those tasks instead of improving the agent system.
**Decision:** Failure context contains only aggregate error-type percentages and
agent attribution. Task names, inputs, and specific errors are never passed to
the evolution LLM.

### One File at a Time (Not All 17 at Once)
**Problem:** Generating all 17 files in one prompt = ~50,000 tokens.
Groq free tier: 30,000 TPM limit → would fail.
**Decision:** One LLM call per file (~2,500 tokens each), with a 15-second delay
between calls to stay within the 30,000 TPM budget.

### LangGraph Instead of Custom Loops
**Problem:** Custom `while` loops for critic/debug cycles are fragile, hard to
extend, and make topology changes require rewriting orchestration logic.
**Decision:** LangGraph `StateGraph` makes topology explicit — adding/removing
agents is a config change, not a code change. Conditional edges replace loops.
`stream_mode="updates"` enables live dashboard phase callbacks.

### ChatGroq Instead of urllib
**Problem:** Manual HTTP calls to Groq needed custom retry logic, rate-limit
handling, connection pooling, and error parsing.
**Decision:** `ChatGroq` from `langchain-groq` handles all of this with
`max_retries=7` and exponential backoff. Code is shorter and more reliable.

### Spawner: Hard Abort Instead of Silent Copy
**Problem:** Old spawner silently copied parent files when the LLM skipped them.
Result: generations were clones — they ran but never evolved.
**Decision:** If required files are missing → `RuntimeError`. The entire
evolution attempt fails rather than creating a partially-cloned generation.

### llm_client.py as Infrastructure
**Problem:** If the LLM regenerates `llm_client.py`, it might revert to urllib
(losing retries), break token tracking, or use a different API.
**Decision:** `llm_client.py` is always copied from parent, never regenerated.
The spawner lists it in `INFRA_FILES` alongside `telemetry.py`.

---

## 17. What Each Generation Can Change

### Can Change (LLM-generated)
- Agent prompts — reasoning strategies, chain-of-thought instructions, restrictions
- Pipeline topology — which agents are enabled, cycle limits
- Algorithm choices — how each agent approaches its task
- File structure — new agents can be added (though currently fixed at 17 files)
- Error handling — how agents handle failures
- Analysis logic — how failures are classified and attributed
- Evolution strategy — how the next generation is designed

### Cannot Change (Infrastructure — copied from parent)
- `llm_client.py` — ChatGroq connection, token tracking, retry logic
- `telemetry.py` — HuggingFace Dataset upload for persistence
- `lineage_memory.py` — cross-generation memory
- `persist.py` — artifact persistence to HF Dataset
- `problem_pool.json` — the 20 benchmark tasks (same for all generations)

---

## 18. File Reference

### Project Root
| File | Purpose |
|---|---|
| `serve.py` | HTTP server + evolution launcher |
| `MISSION_CONTROL.html` | Live dashboard |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for HF Spaces |
| `data/` | Runtime JSON files (heartbeat, evolution log, etc.) |
| `generations/` | One subdirectory per generation |

### `generations/gen_1/` (Seed Generation)
| File | Purpose |
|---|---|
| `main.py` | Entry point — orchestrates task loop, evolution, spawning |
| `config.py` | `Config`, `LLMConfig`, `SandboxConfig`, `AgentTopologyConfig` |
| `pipeline.py` | `AgentPipeline` — LangGraph StateGraph, all node factories |
| `contracts.py` | Typed data contracts between agents |
| `agent_base.py` | `SpecialistAgent` base class |
| `llm_client.py` | `LLMClient` — ChatGroq wrapper with per-agent token tracking |
| `task_manager.py` | `TaskLoader` — loads `problem_pool.json` |
| `analysis.py` | `FailureAnalyzer` — classifies errors, attributes to agents |
| `evolution_engine.py` | `EvolutionEngine` — designs next generation (plan + per-file) |
| `spawner.py` | `Spawner` — validates, writes, launches next generation |
| `agent_analyst.py` | Decomposes problem into `AnalysisSpec` |
| `agent_architect.py` | Designs algorithm as `DesignSpec` |
| `agent_coder.py` | Writes Python code as `CodeArtifact` |
| `agent_critic.py` | Reviews code, produces `CritiqueReport` |
| `agent_reviser.py` | Applies Critic's fixes, produces revised `CodeArtifact` |
| `agent_test_writer.py` | Generates extra test cases as `TestSuite` |
| `agent_debugger.py` | Patches runtime failures, produces patched `CodeArtifact` |
| `agent_clone_inspector.py` | Semantic originality check of offspring files |
| `evolution_tracker.py` | Records spawn events to `data/evolution_log.json` |
| `telemetry.py` | Uploads generation artifacts to HF Dataset for persistence |
| `lineage_memory.py` | Stores cross-generation memory (pass rates, error history) |
| `problem_pool.json` | 20 benchmark tasks (10 SWE-bench + 10 LiveCodeBench) |

---

*Document last updated: 2026-05-25*
*System version: LangGraph pipeline, ChatGroq client, 3-layer anti-clone guardrails*
