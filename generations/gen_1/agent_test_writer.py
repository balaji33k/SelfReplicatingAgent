"""
agent_test_writer.py — TestWriter specialist agent.

Responsibility: Write test cases from the spec alone. Never from the code.
Forbidden from: Seeing CodeArtifact, DesignSpec, or CritiqueReport.

Information contract:
  Receives: AnalysisSpec ONLY
  Produces: TestSuite
  Does NOT receive: any code artifact (prevents confirmation bias)

This isolation is structural, not conventional.
A TestWriter that sees the code would write tests that pass the code,
not tests that verify the spec.
"""
import logging
from typing import Any, Dict, List

from agent_base import SpecialistAgent
from contracts import AnalysisSpec, TestSuite
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class TestWriterAgent(SpecialistAgent):
    """
    Derives test cases purely from the specification.
    Never sees any implementation — writes tests against the spec contract,
    not against what the code happens to do.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("test_writer", llm_client)

    def run(self, analysis: AnalysisSpec) -> TestSuite:  # type: ignore[override]
        self._log_start({"analysis": analysis})

        prompt = self._build_prompt(analysis)
        response = self._call_llm(prompt)
        suite = self._parse_response(analysis.task_id, response)

        self._log_end(suite)
        return suite

    def _build_prompt(self, spec: AnalysisSpec) -> str:
        constraints = "\n".join(f"  - {c}" for c in spec.constraints) or "  - None specified"
        edge_cases = "\n".join(f"  - {e}" for e in spec.edge_cases) or "  - None identified"

        return f"""You are a test engineer. Your only job is to write test cases from the specification.
You have NOT seen any code. You derive tests from the spec contract only.
This prevents you from writing tests that merely confirm what the code does.

SPECIFICATION:
  Problem type: {spec.problem_type}
  Input: {spec.input_description}  [{', '.join(spec.input_types)}]
  Output: {spec.output_description}  [{spec.output_type}]
  Success criteria: {spec.success_criteria}

CONSTRAINTS (must be tested):
{constraints}

EDGE CASES (must each have a test):
{edge_cases}

Write a comprehensive test suite. Cover:
  1. Normal cases — typical valid inputs with known correct outputs
  2. Edge cases — every edge case listed above
  3. Boundary cases — minimum/maximum values, empty collections, single elements
  4. Constraint tests — inputs that approach constraint limits

Produce exactly this format:

NORMAL CASES:
```python
# Each case: (input_args_as_tuple, expected_output)
NORMAL = [
    ((arg1, arg2), expected),
    ...
]
```

EDGE CASES:
```python
EDGE = [
    ((arg1,), expected),
    ...
]
```

BOUNDARY CASES:
```python
BOUNDARY = [
    ((arg1,), expected),
    ...
]
```

CONSTRAINT TESTS:
```python
CONSTRAINTS = [
    ((arg1,), expected),
    ...
]
```

TEST RUNNER:
```python
def run_tests(func):
    all_cases = NORMAL + EDGE + BOUNDARY + CONSTRAINTS
    passed = 0
    failed = 0
    errors = []
    for args, expected in all_cases:
        try:
            result = func(*args)
            if result == expected:
                passed += 1
            else:
                failed += 1
                errors.append(f"Input {{args}} → got {{result!r}}, expected {{expected!r}}")
        except Exception as e:
            failed += 1
            errors.append(f"Input {{args}} → raised {{type(e).__name__}}: {{e}}")
    return passed, failed, errors
```

Write real, runnable Python. Use concrete values for inputs and outputs.
Do NOT write placeholder comments like "# add more tests here".
Every test must have a definite expected output."""

    def _parse_response(self, task_id: str, response: str) -> TestSuite:
        """
        Extract the four test list code blocks and the runner.
        Parses each NORMAL/EDGE/BOUNDARY/CONSTRAINTS block as a Python list literal.
        Falls back to raw code if parsing fails.
        """
        import ast

        def extract_block(label: str) -> List[Any]:
            """Find ```python block under label and eval the list literal."""
            # Find the section
            start = response.find(f"{label}:")
            if start == -1:
                return []
            # Find the next ```python block after the label
            block_start = response.find("```python", start)
            if block_start == -1:
                return []
            block_end = response.find("```", block_start + 9)
            if block_end == -1:
                return []
            block = response[block_start + 9 : block_end].strip()
            # Find the list literal (e.g. "NORMAL = [...]")
            list_match = None
            for line in block.splitlines():
                stripped = line.strip()
                if "=" in stripped and "[" in stripped:
                    list_match = stripped
                    break
            if not list_match:
                return []
            # Extract value after "="
            eq_idx = list_match.index("=")
            list_str = list_match[eq_idx + 1 :].strip()
            # Might span multiple lines — collect until we balance brackets
            # Reconstruct from the block
            eq_pos_in_block = block.find("=")
            if eq_pos_in_block == -1:
                return []
            list_code = block[eq_pos_in_block + 1 :].strip()
            try:
                return ast.literal_eval(list_code)
            except Exception:
                return []

        def extract_runner() -> str:
            """Extract the TEST RUNNER block as raw source."""
            start = response.find("TEST RUNNER:")
            if start == -1:
                return ""
            block_start = response.find("```python", start)
            if block_start == -1:
                return ""
            block_end = response.find("```", block_start + 9)
            if block_end == -1:
                return ""
            return response[block_start + 9 : block_end].strip()

        normal = extract_block("NORMAL CASES")
        edge = extract_block("EDGE CASES")
        boundary = extract_block("BOUNDARY CASES")
        constraint = extract_block("CONSTRAINT TESTS")
        runner = extract_runner()

        all_cases: List[Dict[str, Any]] = []
        for args, expected in normal:
            all_cases.append({"args": list(args), "expected": expected, "category": "normal"})
        for args, expected in edge:
            all_cases.append({"args": list(args), "expected": expected, "category": "edge"})
        for args, expected in boundary:
            all_cases.append({"args": list(args), "expected": expected, "category": "boundary"})
        for args, expected in constraint:
            all_cases.append({"args": list(args), "expected": expected, "category": "constraint"})

        coverage_parts = []
        if normal:
            coverage_parts.append(f"{len(normal)} normal cases")
        if edge:
            coverage_parts.append(f"{len(edge)} edge cases")
        if boundary:
            coverage_parts.append(f"{len(boundary)} boundary cases")
        if constraint:
            coverage_parts.append(f"{len(constraint)} constraint tests")

        coverage_notes = (
            ", ".join(coverage_parts) if coverage_parts else "no structured cases parsed"
        )

        return TestSuite(
            task_id=task_id,
            extra_tests=all_cases,
            coverage_notes=coverage_notes,
        )
