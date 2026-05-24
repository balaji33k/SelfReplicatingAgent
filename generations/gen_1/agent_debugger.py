"""
agent_debugger.py — Debugger specialist agent.

Responsibility: Fix code that fails execution. Pinpoint the crash, patch it.
Forbidden from: Redesigning the algorithm, changing passing test logic,
                inventing new requirements.

Information contract:
  Receives: CodeArtifact + ExecutionContract (execution result with error details)
  Produces: CodeArtifact (patched)
  Does NOT receive: AnalysisSpec or DesignSpec (debugs what failed, not what was planned)
"""
import logging
import re

from agent_base import SpecialistAgent
from contracts import CodeArtifact, ExecutionContract
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class DebuggerAgent(SpecialistAgent):
    """
    Receives code that crashed or produced wrong output during execution.
    Identifies the exact failure point and applies a targeted patch.

    The Debugger does NOT:
      - Redesign the algorithm
      - Change code paths that weren't involved in the failure
      - Add unrelated functionality

    If the code has a logic flaw that requires redesign, it returns the
    original code unchanged and sets `debuggable=False` so the pipeline
    can escalate back to the Architect.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("debugger", llm_client)

    def run(  # type: ignore[override]
        self, artifact: CodeArtifact, execution: ExecutionContract
    ) -> CodeArtifact:
        self._log_start({"artifact": artifact, "execution": execution})

        prompt = self._build_prompt(artifact, execution)
        response = self._call_llm(prompt)
        patched = self._parse_response(artifact.task_id, response, artifact)

        self._log_end(patched)
        return patched

    def _build_prompt(self, artifact: CodeArtifact, execution: ExecutionContract) -> str:
        # Summarise what failed
        failure_lines = []
        for res in execution.test_results:
            if not res.get("passed"):
                inp = res.get("input", "?")
                expected = res.get("expected", "?")
                got = res.get("actual", "?")
                err = res.get("error", "")
                if err:
                    failure_lines.append(f"  Input {inp!r} → ERROR: {err}")
                else:
                    failure_lines.append(
                        f"  Input {inp!r} → got {got!r}, expected {expected!r}"
                    )

        failures_text = "\n".join(failure_lines[:10]) or "  (no specific failures recorded)"
        total_pass = sum(1 for r in execution.test_results if r.get("passed"))
        total = len(execution.test_results)

        return f"""You are a debugger. Your job is to fix a specific runtime failure.
Do NOT redesign the algorithm. Patch only what is broken.

CODE THAT FAILED:
```python
{artifact.code}
```

EXECUTION RESULTS: {total_pass}/{total} tests passed

FAILING CASES:
{failures_text}

STDERR OUTPUT:
{execution.stderr or "(none)"}

ERROR TYPE:
{execution.error_type or "(none)"}

DEBUGGING RULES:
  1. Find the exact line(s) causing the failure
  2. Apply a minimal patch — change as few lines as possible
  3. Do NOT change algorithm structure or working code paths
  4. If the bug requires a full redesign (e.g., the algorithm is fundamentally wrong),
     respond with exactly: NEEDS REDESIGN — then explain why in one sentence.

If the bug is fixable with a patch, respond with the corrected Python function only.
Wrap your code in ```python ... ``` fences."""

    def _parse_response(
        self, task_id: str, response: str, original: CodeArtifact
    ) -> CodeArtifact:
        # Check if the debugger flagged it as needing redesign
        if "NEEDS REDESIGN" in response.upper():
            self.logger.warning(
                f"[debugger] Flagged as NEEDS REDESIGN — returning original with escalation flag"
            )
            # Return original but annotate it needs redesign via task_id suffix
            return CodeArtifact(
                task_id=f"{original.task_id}::needs_redesign",
                code=original.code,
                function_name=original.function_name,
            )

        code = self._extract_code_block(response)

        if not code or len(code.strip()) < 10:
            self.logger.warning(
                f"[debugger] Empty response — keeping original code"
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
