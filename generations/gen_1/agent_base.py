"""
agent_base.py — Base class for all specialist agents.

Every agent:
  - Has one cognitive responsibility
  - Receives a typed input contract
  - Produces a typed output contract
  - Owns its own prompt internally
  - Logs its inputs and outputs for agent-level failure analysis
"""
import logging
import re
from typing import Any, Dict

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class SpecialistAgent:
    """
    Base class for all specialist agents in the pipeline.

    Subclasses must implement:
      run(**inputs) -> typed output

    Each subclass defines:
      - What inputs it accepts (enforced via run() signature)
      - What it produces (enforced via return type)
      - Its own internal prompt (never shared via a central prompts.py)
    """

    def __init__(self, agent_id: str, llm_client: LLMClient):
        self.agent_id = agent_id
        self.llm = llm_client
        self.logger = logging.getLogger(f"Agent.{agent_id}")

    # ── Must override ─────────────────────────────────────────────────────────

    def run(self, **inputs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    # ── LLM call with logging ─────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        self.logger.debug(f"Calling LLM — prompt length: {len(prompt)} chars")
        response = self.llm.call(prompt)
        self.logger.debug(f"LLM response — length: {len(response)} chars")
        return response

    # ── Code cleaning ─────────────────────────────────────────────────────────

    def _strip_markdown(self, text: str) -> str:
        """Remove ```python / ``` fences that LLMs wrap code in."""
        lines = text.splitlines()
        result = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            result.append(line)
        return "\n".join(result)

    def _extract_code_block(self, text: str) -> str:
        """
        Extract the first ```python ... ``` block from a response.
        Falls back to the full text stripped of fences.
        """
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return self._strip_markdown(text).strip()

    # ── Structured output extraction ──────────────────────────────────────────

    def _extract_list(self, text: str, header: str) -> list:
        """
        Extract a bullet list under a given header from LLM output.
        e.g. _extract_list(text, "Edge Cases:") → ["empty list", "negatives"]
        """
        items = []
        in_section = False
        for line in text.splitlines():
            if header.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if not stripped:
                    continue
                # Stop at next header-like line
                if stripped.endswith(":") and len(stripped) < 60 and not stripped.startswith("-"):
                    break
                if stripped.startswith(("-", "*", "•", "+")):
                    items.append(stripped.lstrip("-*•+ ").strip())
        return items

    def _extract_section(self, text: str, header: str, end_header: str = None) -> str:
        """Extract text between two headers."""
        lines = text.splitlines()
        result = []
        in_section = False
        for line in lines:
            if header.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if end_header and end_header.lower() in line.lower():
                    break
                # Stop at any new ALL-CAPS header
                if line.strip().isupper() and len(line.strip()) > 4 and line.strip().endswith(":"):
                    break
                result.append(line)
        return "\n".join(result).strip()

    # ── Input/output logging for failure analysis ─────────────────────────────

    def _log_start(self, inputs: Dict[str, Any]) -> None:
        input_summary = {k: type(v).__name__ for k, v in inputs.items()}
        self.logger.info(f"[{self.agent_id}] START — inputs: {input_summary}")

    def _log_end(self, output: Any, success: bool = True) -> None:
        status = "OK" if success else "FAIL"
        self.logger.info(f"[{self.agent_id}] {status} — output type: {type(output).__name__}")
