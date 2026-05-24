"""
agent_critic.py — Critic specialist agent.

Responsibility: Adversarially review code against the spec. Find bugs.
Forbidden from: Rewriting the code. Receiving DesignSpec (avoids design bias).

Information contract:
  Receives: CodeArtifact + AnalysisSpec
  Produces: CritiqueReport
  Does NOT receive: DesignSpec (no design bias — reviews against spec, not design intent)
"""
import logging

from agent_base import SpecialistAgent
from contracts import AnalysisSpec, CodeArtifact, CritiqueReport
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(SpecialistAgent):
    """
    Adversarially reviews code against the original spec.
    The Critic has no knowledge of HOW the code was designed —
    only what the spec requires and whether the code meets it.
    This eliminates design-bias from the review.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("critic", llm_client)

    def run(self, artifact: CodeArtifact, analysis: AnalysisSpec) -> CritiqueReport:  # type: ignore[override]
        self._log_start({"artifact": artifact, "analysis": analysis})

        prompt = self._build_prompt(artifact, analysis)
        response = self._call_llm(prompt)
        report = self._parse_response(artifact.task_id, response)

        self._log_end(report)
        return report

    def _build_prompt(self, artifact: CodeArtifact, spec: AnalysisSpec) -> str:
        constraints = "\n".join(f"  - {c}" for c in spec.constraints) or "  - None specified"
        edge_cases = "\n".join(f"  - {e}" for e in spec.edge_cases) or "  - None identified"

        return f"""You are an adversarial code reviewer. Your job is to find bugs — not fix them.
You have NOT seen how this code was designed. You review it purely against the specification.

SPECIFICATION:
  Problem type: {spec.problem_type}
  Input: {spec.input_description}  [{', '.join(spec.input_types)}]
  Output: {spec.output_description}  [{spec.output_type}]
  Success criteria: {spec.success_criteria}

CONSTRAINTS (must all be satisfied):
{constraints}

EDGE CASES (must all be handled):
{edge_cases}

CODE TO REVIEW:
```python
{artifact.code}
```

Perform an adversarial review. Look for:
  1. Logic errors — does the algorithm produce wrong results for valid inputs?
  2. Edge case failures — does it break on empty, negative, overflow, duplicate inputs?
  3. Spec deviations — does it return the wrong type? Wrong structure? Wrong semantics?
  4. Off-by-one errors, boundary conditions, index errors
  5. Incorrect handling of the constraints listed above

Produce exactly these sections:

VERDICT:
(PASS if no issues found, FAIL if any issues found)

SEVERITY:
(low / medium / high — based on how many inputs would fail)

LOGIC ERRORS:
- (each logic error on a separate bullet, or "None")

EDGE CASES MISSED:
- (each unhandled edge case on a separate bullet, or "None")

SPEC DEVIATIONS:
- (each deviation from the specification, or "None")

SUGGESTED FIXES:
- (targeted fix description for each issue found — NOT a rewrite, just what to change)

Be rigorous. A false negative (missing a real bug) is worse than a false positive."""

    def _parse_response(self, task_id: str, response: str) -> CritiqueReport:
        verdict_text = self._extract_section(response, "VERDICT:").strip().upper()
        passed = "PASS" in verdict_text and "FAIL" not in verdict_text

        severity_text = self._extract_section(response, "SEVERITY:").strip().lower()
        severity = "medium"
        for s in ("low", "medium", "high"):
            if s in severity_text:
                severity = s
                break

        logic_errors = self._extract_list(response, "LOGIC ERRORS:")
        logic_errors = [e for e in logic_errors if e.lower() != "none"]

        edge_missed = self._extract_list(response, "EDGE CASES MISSED:")
        edge_missed = [e for e in edge_missed if e.lower() != "none"]

        spec_devs = self._extract_list(response, "SPEC DEVIATIONS:")
        spec_devs = [e for e in spec_devs if e.lower() != "none"]

        fixes = self._extract_list(response, "SUGGESTED FIXES:")
        fixes = [f for f in fixes if f.lower() != "none"]

        # If any issues found, override PASS verdict
        has_issues = bool(logic_errors or edge_missed or spec_devs)
        if has_issues:
            passed = False

        return CritiqueReport(
            task_id=task_id,
            passed=passed,
            logic_errors=logic_errors,
            edge_cases_missed=edge_missed,
            spec_deviations=spec_devs,
            suggested_fixes=fixes,
            severity=severity,
        )
