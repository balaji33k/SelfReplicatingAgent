"""
agent_analyst.py — Analyst specialist agent.

Responsibility: Understand the problem. Produce a structured AnalysisSpec.
Forbidden from: Writing any code or suggesting any algorithm.

Information contract:
  Receives: TaskContract (raw problem + test cases)
  Produces: AnalysisSpec
  Does NOT pass to next agent: raw problem statement or test cases
"""
import logging
from typing import List

from agent_base import SpecialistAgent
from contracts import AnalysisSpec, TaskContract
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(SpecialistAgent):
    """
    Decomposes the problem into a structured specification.
    The Architect and TestWriter both depend on this output.
    This agent never writes code — it only reads and reasons about requirements.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("analyst", llm_client)

    def run(self, task: TaskContract) -> AnalysisSpec:  # type: ignore[override]
        self._log_start({"task": task})

        prompt = self._build_prompt(task)
        response = self._call_llm(prompt)
        spec = self._parse_response(task.task_id, response)

        self._log_end(spec)
        return spec

    # Max chars of the raw problem statement to include in the prompt.
    # xarray / SWE-bench issues can be 30k+ chars (≈8k tokens) which exhausts
    # the entire daily token budget on the first run.  3,000 chars ≈ 750 tokens —
    # enough context for the agent to understand the problem without burning budget.
    _MAX_PROBLEM_CHARS = 3000

    def _build_prompt(self, task: TaskContract) -> str:
        test_preview = ""
        if task.test_cases:
            sample = task.test_cases[:3]
            test_preview = f"\nExample test cases (for understanding only — do not solve):\n"
            for i, tc in enumerate(sample, 1):
                test_preview += f"  {i}. Input: {tc.get('input')}  →  Expected: {tc.get('expected')}\n"

        sig_hint = f"\nFunction signature: {task.signature}" if task.signature else ""

        # Truncate very long problem statements (e.g. SWE-bench GitHub issues)
        # to stay within Groq's per-day token budget.
        problem = task.problem_statement or ""
        if len(problem) > self._MAX_PROBLEM_CHARS:
            problem = (
                problem[: self._MAX_PROBLEM_CHARS]
                + f"\n[...truncated: original was {len(task.problem_statement):,} chars]"
            )

        return f"""You are a requirements analyst. Your only job is to deeply understand this problem.
You must NOT write any code or suggest any algorithm. Only analyse and document requirements.

PROBLEM:
{problem}
{sig_hint}
{test_preview}

Produce a structured analysis with exactly these sections:

PROBLEM TYPE:
(one of: algorithm, bug_fix, data_transform, string_processing, math, unknown)

INPUT DESCRIPTION:
(plain English — what inputs does the function receive?)

OUTPUT DESCRIPTION:
(plain English — what must the function return?)

INPUT TYPES:
- (each type on a separate bullet)

OUTPUT TYPE:
(single type)

CONSTRAINTS:
- (each constraint on a separate bullet — size limits, time complexity requirements, restrictions)

EDGE CASES:
- (each edge case on a separate bullet — empty inputs, negatives, duplicates, overflow, etc.)

SUCCESS CRITERIA:
(one sentence — what does a correct solution do?)

AMBIGUITIES:
- (anything unclear in the problem statement, or "None" if clear)

Be thorough. The quality of every downstream agent depends on your analysis."""

    def _parse_response(self, task_id: str, response: str) -> AnalysisSpec:
        def extract(header: str) -> str:
            return self._extract_section(response, header).strip()

        def extract_list(header: str) -> List[str]:
            return self._extract_list(response, header)

        return AnalysisSpec(
            task_id=task_id,
            problem_type=extract("PROBLEM TYPE:").split("\n")[0].strip().lower() or "unknown",
            input_description=extract("INPUT DESCRIPTION:"),
            output_description=extract("OUTPUT DESCRIPTION:"),
            input_types=extract_list("INPUT TYPES:") or ["unknown"],
            output_type=extract("OUTPUT TYPE:").split("\n")[0].strip() or "unknown",
            constraints=extract_list("CONSTRAINTS:"),
            edge_cases=extract_list("EDGE CASES:"),
            success_criteria=extract("SUCCESS CRITERIA:").split("\n")[0].strip(),
            ambiguities=extract_list("AMBIGUITIES:"),
        )
