"""
contracts.py — Typed data contracts between specialist agents.

Every object passed between agents is defined here.
No logic — pure data structures.
Each agent receives only what it needs — information isolation is enforced
by the pipeline, not just convention.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Input ─────────────────────────────────────────────────────────────────────

@dataclass
class TaskContract:
    """
    Normalised view of a benchmark task passed into the pipeline.
    The Analyst is the only agent that receives this directly.
    """
    task_id: str
    problem_statement: str
    test_cases: List[Dict[str, Any]]          # [{"input": ..., "expected": ...}]
    signature: Optional[str] = None           # e.g. "def two_sum(nums: List[int], target: int) -> List[int]"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Stage 1: Analyst → Architect ──────────────────────────────────────────────

@dataclass
class AnalysisSpec:
    """
    Structured decomposition of a problem produced by the Analyst.
    The Architect receives this — never the raw problem statement.
    The TestWriter receives ONLY this — never the code.
    """
    task_id: str
    problem_type: str                         # "algorithm" | "bug_fix" | "data_transform" | "unknown"
    input_description: str                    # plain English description of inputs
    output_description: str                   # plain English description of expected output
    input_types: List[str]                    # e.g. ["List[int]", "int"]
    output_type: str                          # e.g. "List[int]"
    constraints: List[str]                    # e.g. ["1 <= len(nums) <= 10^4", "no built-in sort"]
    edge_cases: List[str]                     # e.g. ["empty list", "all negatives", "duplicates"]
    success_criteria: str                     # one sentence defining what "correct" means
    ambiguities: List[str] = field(default_factory=list)  # things that are unclear


# ── Stage 2: Architect → Coder ────────────────────────────────────────────────

@dataclass
class DesignSpec:
    """
    Algorithm design produced by the Architect.
    The Coder receives this — never the raw problem statement or test cases.
    """
    task_id: str
    algorithm_name: str                       # e.g. "Two-pointer scan"
    algorithm_rationale: str                  # why this algorithm fits the constraints
    data_structures: List[str]                # e.g. ["hash map", "deque"]
    steps: List[str]                          # numbered pseudocode steps
    time_complexity: str                      # e.g. "O(n)"
    space_complexity: str                     # e.g. "O(1)"
    edge_case_strategies: Dict[str, str]      # edge_case_description → how to handle it
    assumptions: List[str] = field(default_factory=list)  # what the design assumes


# ── Stage 3: Coder → Critic ───────────────────────────────────────────────────

@dataclass
class CodeArtifact:
    """
    Python code produced by the Coder.
    The Critic receives this alongside AnalysisSpec — NOT DesignSpec.
    This prevents the Critic from being biased by the design choices.
    """
    task_id: str
    code: str                                 # clean Python, fences stripped
    function_name: str                        # extracted from code or signature


# ── Stage 4: Critic → Reviser ─────────────────────────────────────────────────

@dataclass
class CritiqueReport:
    """
    Adversarial review produced by the Critic.
    The Reviser receives this alongside CodeArtifact.
    """
    task_id: str
    passed: bool                              # True = no issues found, skip Reviser
    logic_errors: List[str]                   # e.g. ["off-by-one in boundary check"]
    edge_cases_missed: List[str]              # e.g. ["empty list not handled"]
    spec_deviations: List[str]                # e.g. ["returns int instead of List[int]"]
    suggested_fixes: List[str]                # targeted fix descriptions (not rewrites)
    severity: str = "low"                     # "low" | "medium" | "high"


# ── Stage 6: TestWriter (from AnalysisSpec only) ──────────────────────────────

@dataclass
class TestSuite:
    """
    Additional test cases produced by the TestWriter.
    The TestWriter receives ONLY AnalysisSpec — never the code.
    This guarantees implementation-independent testing.
    """
    task_id: str
    extra_tests: List[Dict[str, Any]]         # same format as task.test_cases
    coverage_notes: str                       # what boundaries / edges these cover


# ── Stage 7: Executor → Debugger ─────────────────────────────────────────────

@dataclass
class ExecutionContract:
    """
    Result of sandbox execution passed to the Debugger on failure.
    The Debugger receives this alongside CodeArtifact — NOT DesignSpec or AnalysisSpec.
    The Debugger diagnoses from the error alone, not from the original design.
    """
    task_id: str
    success: bool
    stdout: str
    stderr: str
    error_type: str                           # "SyntaxError" | "RuntimeError" | "LogicError" | "TimeoutError"
    runtime: float
    test_results: List[Dict[str, Any]] = field(default_factory=list)


# ── Final output ──────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete result of the agent pipeline for one task.
    Superset of ExecutionContract — adds agent provenance data.
    """
    task_id: str
    success: bool
    final_code: str
    error_type: Optional[str]
    runtime: float
    stdout: str = ""
    stderr: str = ""

    # Agent provenance — used by FailureAnalyzer for per-agent attribution
    agents_used: List[str] = field(default_factory=list)
    cycles: Dict[str, int] = field(default_factory=dict)      # {"reviser": 1, "debugger": 0}
    agent_failures: Dict[str, str] = field(default_factory=dict)  # {"critic": "missed_logic"}

    # Phase-level outcomes — None=not reached, True=passed, False=failed
    # Keys: "compile", "unit_tests", "integration_tests"
    phase_results: Dict[str, Optional[bool]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_code": self.final_code,
            "error_type": self.error_type,
            "runtime": self.runtime,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "agents_used": self.agents_used,
            "cycles": self.cycles,
            "agent_failures": self.agent_failures,
            "phase_results": self.phase_results,
        }
