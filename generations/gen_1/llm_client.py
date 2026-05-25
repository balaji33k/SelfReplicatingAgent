"""
llm_client.py — LangChain/Groq-backed LLM client with per-agent token tracking.

Uses ChatGroq from langchain-groq for all LLM calls.
Exposes the same .call() / .chat_completion() interface that every agent
in agent_base.py uses — no agent files need to change.

Token tracking:
  Every call records input + output tokens against the calling agent name.
  Counts are written to data/token_usage.json after each call so the dashboard
  can display: model name, daily budget remaining, and per-agent usage.

Rate-limit handling (two-tier):
  1. TPM (per-minute) limit — sleep for the exact retry time in the error message,
     then retry the same call.  Maximum sleep: 15 minutes.
  2. TPD (per-day) limit — when retry time > 15 min, switch to the next fallback
     model in _FALLBACK_ORDER (each has its own independent 500k TPD budget).
     If all fallback models are exhausted, raise the error.

Provider: Groq (GROQ_API_KEY).
Model: set in config.py (default: meta-llama/llama-4-scout-17b-16e-instruct).
"""
import datetime
import json
import logging
import os
import re
import threading
import time
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

# Ordered fallback chain — when primary hits TPD, switch to next model.
# Each model has its own independent TPD budget at Groq free tier.
_FALLBACK_ORDER = [
    "meta-llama/llama-4-scout-17b-16e-instruct",   # primary (30k TPM / 500k TPD)
    "meta-llama/llama-4-maverick-17b-128e-instruct", # fallback 1 (30k TPM / 500k TPD)
    "llama-3.1-8b-instant",                           # fallback 2 (20k TPM / 500k TPD)
    "gemma2-9b-it",                                    # fallback 3 (15k TPM / 500k TPD)
    "mixtral-8x7b-32768",                              # fallback 4 (18k TPM / 500k TPD)
]

# Max wait time (seconds) for a TPM rate-limit sleep.
# If the error says "retry in > N seconds", treat it as TPD exhaustion and switch models.
_MAX_TPM_SLEEP_SEC = 900   # 15 minutes


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
            max_retries=2,   # only for transient network errors; 429 handled below
        )
        self._api_key = api_key
        # Track which fallback models have already been exhausted this session
        self._exhausted_models: set = set()

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

        Handles Groq rate limits automatically:
          - TPM (per-minute): sleep for the retry duration, then retry.
          - TPD (per-day): switch to next fallback model; retry immediately.

        agent_name: the calling agent's ID (e.g. 'analyst', 'coder').
                    Used for per-agent token tracking in the dashboard.
        """
        logger.debug(f"[llm_client] call() agent={agent_name or '?'} len={len(prompt)}")
        return self._invoke_with_retry([HumanMessage(content=prompt)], agent_name or "unknown")

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

        return self._invoke_with_retry(lc_messages, agent_name or "unknown")

    def _invoke_with_retry(self, lc_messages, agent_name: str) -> str:
        """
        Invoke the LLM with automatic rate-limit handling.

        Strategy:
          1. On 429 with retry_time <= _MAX_TPM_SLEEP_SEC → sleep, same model.
          2. On 429 with retry_time >  _MAX_TPM_SLEEP_SEC → mark model exhausted,
             switch to next fallback, retry immediately.
          3. If no fallbacks remain → re-raise so callers can handle it.
        """
        max_attempts = len(_FALLBACK_ORDER) + 2   # enough for all fallbacks + 2 TPM retries
        for attempt in range(max_attempts):
            try:
                response = self._llm.invoke(lc_messages)
                self._record_usage(agent_name, response)
                return response.content
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = ("429" in err_str or "rate_limit_exceeded" in err_str
                                 or "RateLimitError" in type(exc).__name__)
                if not is_rate_limit:
                    raise

                retry_sec = self._parse_retry_seconds(err_str)
                current_model = self.config.model_name

                if retry_sec <= _MAX_TPM_SLEEP_SEC:
                    # TPM (per-minute) limit — short wait, retry same model
                    sleep_time = retry_sec + 5
                    logger.warning(
                        f"[llm_client] TPM rate limit on {current_model} — "
                        f"sleeping {sleep_time:.0f}s (attempt {attempt+1})"
                    )
                    time.sleep(sleep_time)
                else:
                    # TPD (per-day) exhausted on this model — switch to fallback
                    self._exhausted_models.add(current_model)
                    logger.warning(
                        f"[llm_client] TPD exhausted on {current_model} "
                        f"(retry in {retry_sec:.0f}s) — switching to fallback model"
                    )
                    if not self._switch_to_next_model():
                        # No more fallbacks
                        raise RuntimeError(
                            f"All Groq models exhausted for today. "
                            f"Exhausted: {self._exhausted_models}. "
                            f"Last error: {exc}"
                        ) from exc
        raise RuntimeError(f"[llm_client] Exceeded {max_attempts} retry attempts")

    def _parse_retry_seconds(self, error_msg: str) -> float:
        """
        Parse 'Please try again in Xm Y.Zs' → total seconds.
        Returns a large number (9999) if the pattern is not found.
        """
        # "13m3.8208s" pattern
        m = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", error_msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        # "45.5s" pattern (seconds only)
        m = re.search(r"try again in (\d+(?:\.\d+)?)s", error_msg)
        if m:
            return float(m.group(1))
        return 9999.0

    def _switch_to_next_model(self) -> bool:
        """
        Switch self._llm to the next non-exhausted model in _FALLBACK_ORDER.
        Returns True if a switch was made, False if all models are exhausted.
        """
        for model_name in _FALLBACK_ORDER:
            if model_name not in self._exhausted_models:
                try:
                    self._llm = ChatGroq(
                        model=model_name,
                        temperature=self.config.temperature,
                        api_key=self._api_key,
                        max_tokens=8192,
                        timeout=self.config.timeout_seconds,
                        max_retries=2,
                    )
                    old_model = self.config.model_name
                    self.config.model_name = model_name
                    self._limits = _MODEL_LIMITS.get(model_name, _DEFAULT_LIMITS)
                    logger.info(
                        f"[llm_client] ✓ Switched model: {old_model} → {model_name} "
                        f"(new budget: {self._limits['tpd']:,} TPD / {self._limits['tpm']:,} TPM)"
                    )
                    return True
                except Exception as e:
                    logger.error(f"[llm_client] Failed to switch to {model_name}: {e}")
                    self._exhausted_models.add(model_name)
        return False

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
