"""
agent_coder.py — Coder specialist agent.

Responsibility: Translate the DesignSpec into clean Python. Nothing else.
Forbidden from: Re-analyzing the problem, re-designing the algorithm,
                looking at raw test cases.

Information contract:
  Receives: DesignSpec + AnalysisSpec (for type hints and function name only)
  Produces: CodeArtifact
"""
import re
import logging

from agent_base import SpecialistAgent
from contracts import AnalysisSpec, CodeArtifact, DesignSpec
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class CoderAgent(SpecialistAgent):
    """
    Translates a DesignSpec into Python code.
    Trusts the design — does not second-guess the algorithm.
    Uses AnalysisSpec only for type hints and function signature.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("coder", llm_client)

    def run(self, design: DesignSpec, analysis: AnalysisSpec) -> CodeArtifact:  # type: ignore[override]
        self._log_start({"design": design, "analysis": analysis})

        prompt = self._build_prompt(design, analysis)
        response = self._call_llm(prompt)
        artifact = self._parse_response(design.task_id, analysis, response)

        self._log_end(artifact)
        return artifact

    def _build_prompt(self, design: DesignSpec, spec: AnalysisSpec) -> str:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(design.steps))
        edge_strategies = "\n".join(
            f"  - {k}: {v}" for k, v in design.edge_case_strategies.items()
        ) or "  - None specified"
        ds = ", ".join(design.data_structures) or "standard Python"

        return f"""You are a Python programmer. Your only job is to translate the design below into clean Python.
Do not re-analyse the problem. Do not change the algorithm. Just implement it precisely.

DESIGN TO IMPLEMENT:
  Algorithm: {design.algorithm_name}
  Rationale: {design.algorithm_rationale}
  Data structures: {ds}
  Time complexity: {design.time_complexity}
  Space complexity: {design.space_complexity}

IMPLEMENTATION STEPS (follow exactly in order):
{steps}

EDGE CASE HANDLING (must be in your implementation):
{edge_strategies}

FUNCTION SPECIFICATION:
  Input types: {', '.join(spec.input_types)}
  Output type: {spec.output_type}
  Success criteria: {spec.success_criteria}

REQUIREMENTS:
  - Write a single self-contained Python function
  - Use only Python standard library — no third-party imports
  - Handle every edge case listed in the design
  - Add a brief docstring explaining what the function does
  - No print statements, no example usage outside the function
  - The function must be the ONLY top-level definition

Respond with ONLY the Python function. No explanation before or after.
Wrap your code in ```python ... ``` fences."""

    def _parse_response(self, task_id: str, spec: AnalysisSpec, response: str) -> CodeArtifact:
        code = self._extract_code_block(response)

        # Extract function name from code
        func_name = "solve"
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if match:
            func_name = match.group(1)

        return CodeArtifact(
            task_id=task_id,
            code=code,
            function_name=func_name,
        )
