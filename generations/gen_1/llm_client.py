"""
llm_client.py — LangChain/Groq-backed LLM client with per-agent token tracking.

Uses ChatGroq from langchain-groq for all LLM calls.
Exposes the same .call() / .chat_completion() interface that every agent
in agent_base.py uses — no agent files need to change.

Token tracking:
  Every call records input + output tokens against the calling agent name.
  Counts are written to data/token_usage.json after each call so the dashboard
  can display: model name, daily budget remaining, and per-agent usage.

Provider: Groq (GROQ_API_KEY).
Model: set in config.py (default: meta-llama/llama-4-scout-17b-16e-instruct).
Retries: handled automatically by LangChain (max_retries in ChatGroq constructor).
"""
import datetime
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Known Groq free-tier token limits ────────────────────────────────────────
# tpd = tokens per day, tpm = tokens per minute
_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"tpd": 500_000, "tpm": 30_000},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"tpd": 500_000, "tpm": 30_000},
    "llama-3.3-70b-versatile":                   {"tpd": 100_000, "tpm": 12_000},
    "llama-3.1-70b-versatile":                   {"tpd": 100_000, "tpm": 12_000},
    "llama-3.1-8b-instant":                       {"tpd": 500_000, "tpm": 20_000},
    "mixtral-8x7b-32768":                         {"tpd": 500_000, "tpm": 18_000},
    "gemma2-9b-it":                               {"tpd": 500_000, "tpm": 15_000},
}
_DEFAULT_LIMITS = {"tpd": 500_000, "tpm": 30_000}


class LLMClient:
    """
    Thin wrapper around ChatGroq that:
      - Preserves the .call() / .chat_completion() interface used by all agents.
      - Tracks token usage per agent and writes it to data/token_usage.json
        so the dashboard can show daily budget remaining and per-agent breakdown.

    LangChain handles:
      - HTTP connection pooling
      - Automatic retries with exponential backoff (max_retries)
      - Rate-limit error handling
    """

    _usage_lock = threading.Lock()

    def __init__(self, config):
        self.config = config
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")

        self._llm = ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            max_tokens=8192,
            timeout=config.timeout_seconds,
            max_retries=7,
        )

        # ── Token usage tracking (resets daily) ──────────────────────────────
        self._usage_date: str = datetime.date.today().isoformat()
        self._per_agent: Dict[str, Dict] = {}   # {agent_name: {tokens, calls}}
        self._total_tokens: int = 0
        self._limits = _MODEL_LIMITS.get(config.model_name, _DEFAULT_LIMITS)

        # Resolve project root to locate data/ directory
        gen_dir = Path(__file__).resolve().parent
        project_root = (
            gen_dir.parent.parent
            if gen_dir.parent.name == "generations"
            else gen_dir.parent
        )
        self._token_file = project_root / "data" / "token_usage.json"

        logger.info(
            f"[llm_client] ChatGroq initialised — "
            f"model={config.model_name} temperature={config.temperature} "
            f"budget={self._limits['tpd']:,} TPD / {self._limits['tpm']:,} TPM"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def call(self, prompt: str, agent_name: str = "") -> str:
        """
        Send a single user prompt and return the response text.
        agent_name: the calling agent's ID (e.g. 'analyst', 'coder').
                    Used for per-agent token tracking in the dashboard.
        """
        logger.debug(f"[llm_client] call() agent={agent_name or '?'} len={len(prompt)}")
        response = self._llm.invoke([HumanMessage(content=prompt)])
        self._record_usage(agent_name or "unknown", response)
        return response.content

    def chat_completion(
        self,
        messages: List[Dict],
        response_format: Optional[Dict] = None,
        agent_name: str = "",
    ) -> str:
        """
        Multi-turn chat interface.
        response_format is ignored (Groq handles JSON mode separately).
        agent_name: same as in call() — for dashboard token tracking.
        """
        lc_messages = []
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = self._llm.invoke(lc_messages)
        self._record_usage(agent_name or "unknown", response)
        return response.content

    # ── Token tracking ────────────────────────────────────────────────────────

    def _record_usage(self, agent_name: str, response) -> None:
        """
        Extract token counts from the ChatGroq AIMessage response and accumulate.

        langchain-core 0.3+ puts them in response.usage_metadata:
          {"input_tokens": N, "output_tokens": N, "total_tokens": N}
        Groq also sets response.response_metadata["token_usage"]["total_tokens"].
        """
        try:
            tokens = 0

            # Primary: langchain-core standard attribute
            meta = getattr(response, "usage_metadata", None) or {}
            tokens = int(meta.get("total_tokens", 0) or 0)

            if not tokens:
                # Fallback: Groq-specific response_metadata
                rmeta = getattr(response, "response_metadata", None) or {}
                usage = rmeta.get("token_usage", {})
                tokens = int(usage.get("total_tokens", 0) or 0)

            with self._usage_lock:
                today = datetime.date.today().isoformat()
                if today != self._usage_date:
                    # Day rolled over — reset counters
                    self._per_agent = {}
                    self._total_tokens = 0
                    self._usage_date = today

                entry = self._per_agent.setdefault(
                    agent_name, {"tokens": 0, "calls": 0}
                )
                entry["tokens"] += tokens
                entry["calls"]  += 1
                self._total_tokens += tokens

            self._flush_usage()

        except Exception as exc:
            logger.debug(f"[llm_client] token tracking error (non-fatal): {exc}")

    def _flush_usage(self) -> None:
        """Write current token usage to data/token_usage.json for the dashboard."""
        try:
            tpd = self._limits["tpd"]
            tpm = self._limits["tpm"]

            with self._usage_lock:
                data = {
                    "model":               self.config.model_name,
                    "provider":            "groq",
                    "date":                self._usage_date,
                    "daily_limit_tokens":  tpd,
                    "tpm_limit":           tpm,
                    "total_tokens_used":   self._total_tokens,
                    "tokens_remaining":    max(0, tpd - self._total_tokens),
                    "pct_used":            round(self._total_tokens / tpd, 4) if tpd else 0,
                    "per_agent":           dict(self._per_agent),
                    "last_updated":        datetime.datetime.utcnow().isoformat() + "Z",
                }

            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            self._token_file.write_text(json.dumps(data, indent=2))

        except Exception as exc:
            logger.debug(f"[llm_client] token_usage flush error (non-fatal): {exc}")
