"""
agent_reviser.py — Reviser specialist agent.

Responsibility: Apply targeted fixes from the Critic's report. Nothing else.
Forbidden from: Redesigning the algorithm, inventing new requirements,
                changing code that the Critic did not flag.

Information contract:
  Receives: CodeArtifact + CritiqueReport
  Produces: CodeArtifact (revised)
  Does NOT receive: AnalysisSpec, DesignSpec (stays within Critic's scope)
"""
import logging
import re

from agent_base import SpecialistAgent
from contracts import CodeArtifact, CritiqueReport
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReviserAgent(SpecialistAgent):
    """
    Applies exactly the fixes the Critic identified.
    The Reviser is NOT allowed to:
      - Change working code the Critic didn't flag
      - Redesign the algorithm
      - Invent additional requirements

    Bounded by max_cycles to prevent infinite revision loops.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("reviser", llm_client)

    def run(  # type: ignore[override]
        self,
        artifact: CodeArtifact,
        critique: CritiqueReport,
        cycle: int = 1,
    ) -> CodeArtifact:
        self._log_start({"artifact": artifact, "critique": critique, "cycle": cycle})

        prompt = self._build_prompt(artifact, critique, cycle)
        response = self._call_llm(prompt)
        revised = self._parse_response(artifact.task_id, response, artifact)

        self._log_end(revised)
        return revised

    def _build_prompt(
        self, artifact: CodeArtifact, critique: CritiqueReport, cycle: int
    ) -> str:
        logic_errors = "\n".join(f"  - {e}" for e in critique.logic_errors) or "  - None"
        edge_missed = "\n".join(f"  - {e}" for e in critique.edge_cases_missed) or "  - None"
        spec_devs = "\n".join(f"  - {e}" for e in critique.spec_deviations) or "  - None"
        fixes = "\n".join(f"  - {f}" for f in critique.suggested_fixes) or "  - None"

        return f"""You are a surgical code fixer. Revision cycle {cycle}.
Your ONLY job is to apply the exact fixes listed below. Nothing more.

RULES:
  - Fix ONLY what the critic flagged
  - Do NOT redesign the algorithm
  - Do NOT change working code that was not flagged
  - Do NOT add new functionality
  - Preserve the function name and signature exactly

ORIGINAL CODE:
```python
{artifact.code}
```

CRITIC'S REPORT (severity: {critique.severity}):

LOGIC ERRORS to fix:
{logic_errors}

EDGE CASES to handle:
{edge_missed}

SPEC DEVIATIONS to correct:
{spec_devs}

SUGGESTED FIXES (apply these precisely):
{fixes}

Apply the minimum necessary changes to fix each issue.
Respond with ONLY the corrected Python function.
Wrap your code in ```python ... ``` fences."""

    def _parse_response(
        self, task_id: str, response: str, original: CodeArtifact
    ) -> CodeArtifact:
        code = self._extract_code_block(response)

        if not code or len(code.strip()) < 10:
            # If LLM returns garbage, keep original
            self.logger.warning(
                f"[reviser] Empty or trivial response — keeping original code"
            )
            return original

        func_name = original.function_name
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if match:
            func_name = match.group(1)

        return CodeArtifact(
            task_id=task_id,
            code=code,
            function_name=func_name,
        )
