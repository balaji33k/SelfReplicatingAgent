"""
agent_architect.py — Architect specialist agent.

Responsibility: Design the solution strategy. Produce a DesignSpec.
Forbidden from: Writing any Python code or looking at raw test cases.

Information contract:
  Receives: AnalysisSpec
  Produces: DesignSpec
  Does NOT receive: Raw problem statement, test cases
"""
import logging
from typing import Dict, List

from agent_base import SpecialistAgent
from contracts import AnalysisSpec, DesignSpec
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ArchitectAgent(SpecialistAgent):
    """
    Designs the algorithmic solution from the structured analysis.
    Produces a detailed pseudocode plan the Coder will translate directly.
    Never writes Python — only designs strategy.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("architect", llm_client)

    def run(self, analysis: AnalysisSpec) -> DesignSpec:  # type: ignore[override]
        self._log_start({"analysis": analysis})

        prompt = self._build_prompt(analysis)
        response = self._call_llm(prompt)
        design = self._parse_response(analysis.task_id, response)

        self._log_end(design)
        return design

    def _build_prompt(self, spec: AnalysisSpec) -> str:
        constraints = "\n".join(f"  - {c}" for c in spec.constraints) or "  - None specified"
        edge_cases = "\n".join(f"  - {e}" for e in spec.edge_cases) or "  - None identified"

        return f"""You are a solution architect. Your only job is to design the algorithm — not write code.
You must NOT write any Python. Only design the approach in pseudocode and plain English.

PROBLEM ANALYSIS:
  Type: {spec.problem_type}
  Input: {spec.input_description}
  Output: {spec.output_description}
  Input types: {', '.join(spec.input_types)}
  Output type: {spec.output_type}
  Success criteria: {spec.success_criteria}

CONSTRAINTS:
{constraints}

EDGE CASES TO HANDLE:
{edge_cases}

Design the optimal solution. Produce exactly these sections:

ALGORITHM NAME:
(short name for the algorithm or approach)

ALGORITHM RATIONALE:
(why this algorithm is correct and fits the constraints — one paragraph)

DATA STRUCTURES:
- (each data structure on a separate bullet — e.g. hash map, deque, two pointers)

SOLUTION STEPS:
1. (first step — precise and unambiguous)
2. (second step)
... (continue until complete)

TIME COMPLEXITY:
(Big-O notation with brief explanation)

SPACE COMPLEXITY:
(Big-O notation with brief explanation)

EDGE CASE STRATEGIES:
- <edge case>: <how to handle it in the algorithm>
- (one per bullet)

ASSUMPTIONS:
- (anything the design assumes about the input — or "None")

Be precise. The Coder will translate your steps to Python directly without additional reasoning."""

    def _parse_response(self, task_id: str, response: str) -> DesignSpec:
        def extract(header: str) -> str:
            return self._extract_section(response, header).strip()

        def extract_list(header: str) -> List[str]:
            return self._extract_list(response, header)

        def extract_dict(header: str) -> Dict[str, str]:
            items = extract_list(header)
            result = {}
            for item in items:
                if ":" in item:
                    k, v = item.split(":", 1)
                    result[k.strip()] = v.strip()
                else:
                    result[item] = "handle appropriately"
            return result

        # Parse numbered steps
        steps_text = extract("SOLUTION STEPS:")
        steps = []
        for line in steps_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                steps.append(line.lstrip("0123456789.-) ").strip())

        return DesignSpec(
            task_id=task_id,
            algorithm_name=extract("ALGORITHM NAME:").split("\n")[0].strip() or "general approach",
            algorithm_rationale=extract("ALGORITHM RATIONALE:"),
            data_structures=extract_list("DATA STRUCTURES:"),
            steps=steps or [steps_text],
            time_complexity=extract("TIME COMPLEXITY:").split("\n")[0].strip() or "unknown",
            space_complexity=extract("SPACE COMPLEXITY:").split("\n")[0].strip() or "unknown",
            edge_case_strategies=extract_dict("EDGE CASE STRATEGIES:"),
            assumptions=extract_list("ASSUMPTIONS:"),
        )
