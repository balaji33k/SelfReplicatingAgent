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
# These are approximate free-tier limits; actual limits may vary by account.
_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "meta-llama/llama-4-scout-17b-16e-instruct":  {"tpd": 500_000, "tpm": 30_000},
    "llama-3.1-8b-instant":                        {"tpd": 500_000, "tpm": 20_000},
    "llama-3.3-70b-versatile":                     {"tpd": 100_000, "tpm": 12_000},
    "deepseek-r1-distill-llama-70b":               {"tpd": 500_000, "tpm": 30_000},
    "deepseek-r1-distill-qwen-32b":                {"tpd": 500_000, "tpm": 30_000},
    "qwen-qwq-32b":                                {"tpd": 500_000, "tpm": 30_000},
}
_DEFAULT_LIMITS = {"tpd": 500_000, "tpm": 30_000}

# Ordered fallback chain — when primary hits TPD/unavailable, switch to next.
# Each model has its own independent daily token budget at Groq.
# CONFIRMED DECOMMISSIONED (do not add back):
#   - llama4-maverick: 404 (not on free tier)
#   - gemma2-9b-it, mixtral-8x7b-32768, llama3-70b-8192, llama3-8b-8192: decommissioned
_FALLBACK_ORDER = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # primary   (30k TPM / 500k TPD)
    "llama-3.1-8b-instant",                         # fallback1 (20k TPM / 500k TPD)
    "llama-3.3-70b-versatile",                       # fallback2 (12k TPM / 100k TPD)
    "deepseek-r1-distill-llama-70b",                 # fallback3 (newer Groq model)
    "deepseek-r1-distill-qwen-32b",                  # fallback4 (newer Groq model)
    "qwen-qwq-32b",                                  # fallback5 (newer Groq model)
]

# Max wait (seconds) for a per-minute rate limit before sleeping and retrying.
# Anything longer than this is treated as a per-day limit → switch model.
_MAX_TPM_SLEEP_SEC = 900   # 15 minutes

# When ALL fallback models are exhausted, wait this long before trying again
# from the beginning of the chain (rolling 24h window will have freed some tokens).
_RECOVERY_WAIT_SEC = 1800   # 30 minutes
_MAX_RECOVERY_ATTEMPTS = 4   # give up after 4 full-chain retries (= 2 hours total)


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
        # Track which fallback models have already been exhausted this session.
        # Two separate buckets so the dashboard can show different colours.
        self._exhausted_models: set = set()   # merged set used by retry loop
        self._tpd_exhausted: set = set()      # ran out of daily tokens
        self._unavailable: set = set()        # 404 / decommissioned

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
        self._status_file = project_root / "data" / "model_status.json"

        logger.info(
            f"[llm_client] ChatGroq initialised — "
            f"model={config.model_name} temperature={config.temperature} "
            f"budget={self._limits['tpd']:,} TPD / {self._limits['tpm']:,} TPM"
        )
        # Write initial status so the dashboard shows the active model immediately
        self._flush_model_status()

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
        # Outer loop: allow full-chain retries after a recovery wait
        for recovery_attempt in range(_MAX_RECOVERY_ATTEMPTS):
            # Inner loop: try each model in the fallback chain
            for attempt in range(len(_FALLBACK_ORDER) + 4):
                try:
                    response = self._llm.invoke(lc_messages)
                    self._record_usage(agent_name, response)
                    return response.content
                except Exception as exc:
                    err_str = str(exc)
                    is_rate_limit = (
                        "429" in err_str
                        or "rate_limit_exceeded" in err_str
                        or "RateLimitError" in type(exc).__name__
                    )
                    is_unavailable = (
                        "404" in err_str
                        or "model_not_found" in err_str
                        or "does not exist" in err_str
                        or "model_decommissioned" in err_str
                        or "decommissioned" in err_str.lower()
                        or "no longer supported" in err_str.lower()
                    )

                    if is_rate_limit:
                        retry_sec = self._parse_retry_seconds(err_str)
                        current_model = self.config.model_name
                        logger.debug(
                            f"[llm_client] 429 on {current_model} — "
                            f"retry_sec={retry_sec:.0f} | raw_msg={err_str[:200]}"
                        )

                        if retry_sec <= _MAX_TPM_SLEEP_SEC:
                            # TPM (per-minute) limit — sleep and retry same model
                            sleep_time = retry_sec + 5
                            logger.warning(
                                f"[llm_client] TPM rate limit on {current_model} — "
                                f"sleeping {sleep_time:.0f}s (attempt {attempt+1})"
                            )
                            time.sleep(sleep_time)
                        else:
                            # TPD (per-day) exhausted — switch to next fallback
                            self._exhausted_models.add(current_model)
                            self._tpd_exhausted.add(current_model)
                            self._flush_model_status()
                            logger.warning(
                                f"[llm_client] TPD exhausted on {current_model} "
                                f"(retry in {retry_sec:.0f}s) — switching fallback"
                            )
                            if not self._switch_to_next_model():
                                break  # all models exhausted → go to recovery wait

                    elif is_unavailable:
                        bad_model = self.config.model_name
                        self._exhausted_models.add(bad_model)
                        self._unavailable.add(bad_model)
                        self._flush_model_status()
                        logger.warning(
                            f"[llm_client] Model {bad_model} unavailable/decommissioned — "
                            f"switching to next fallback"
                        )
                        if not self._switch_to_next_model():
                            break  # all models exhausted → go to recovery wait
                    else:
                        raise  # real error — propagate immediately
            else:
                # Inner loop finished normally (shouldn't happen) — break outer
                break

            # ── All fallback models exhausted ────────────────────────────────
            # Wait for the Groq rolling 24h window to free up some tokens,
            # then reset and try from the primary model again.
            logger.warning(
                f"[llm_client] All Groq models exhausted "
                f"(tried: {self._exhausted_models}). "
                f"Waiting {_RECOVERY_WAIT_SEC}s for token budget to partially recover "
                f"(recovery attempt {recovery_attempt+1}/{_MAX_RECOVERY_ATTEMPTS})..."
            )
            time.sleep(_RECOVERY_WAIT_SEC)
            # Reset exhausted set — tokens from 30min ago have rolled off the 24h window
            self._exhausted_models.clear()
            if not self._switch_to_next_model():
                # Switch to primary explicitly
                self._switch_to_model(_FALLBACK_ORDER[0])

        raise RuntimeError(
            f"[llm_client] All Groq models exhausted after {_MAX_RECOVERY_ATTEMPTS} "
            f"recovery attempts ({_MAX_RECOVERY_ATTEMPTS * _RECOVERY_WAIT_SEC / 3600:.1f}h total). "
            f"Last tried: {self._exhausted_models}"
        )

    def _parse_retry_seconds(self, error_msg: str) -> float:
        """
        Parse Groq retry-after strings → total seconds.
        Handles all known formats:
          "13m3.8208s"    → 783s
          "8h32m11.2s"   → 30731s
          "8h32m"        → 30720s
          "8h"           → 28800s
          "45.5s"        → 45.5s
        Returns _MAX_TPM_SLEEP_SEC + 1 (forces model switch) if unparseable.
        """
        # "Xh Ym Zs" — hours + minutes + seconds
        m = re.search(r"try again in (\d+)h(\d+)m(\d+(?:\.\d+)?)s", error_msg)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        # "Xh Ym" — hours + minutes only
        m = re.search(r"try again in (\d+)h(\d+)m\b", error_msg)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60
        # "Xh" — hours only
        m = re.search(r"try again in (\d+)h\b", error_msg)
        if m:
            return int(m.group(1)) * 3600
        # "Xm Y.Zs" — minutes + seconds
        m = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", error_msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        # "X.Ys" — seconds only
        m = re.search(r"try again in (\d+(?:\.\d+)?)s", error_msg)
        if m:
            return float(m.group(1))
        # Unknown format — log the raw message and treat as long wait (switch model)
        logger.debug(f"[llm_client] Could not parse retry time from: {error_msg[:300]}")
        return _MAX_TPM_SLEEP_SEC + 1

    def _switch_to_model(self, model_name: str) -> bool:
        """Switch to a specific model by name. Returns True on success."""
        try:
            old_model = self.config.model_name
            self._llm = ChatGroq(
                model=model_name,
                temperature=self.config.temperature,
                api_key=self._api_key,
                max_tokens=8192,
                timeout=self.config.timeout_seconds,
                max_retries=2,
            )
            self.config.model_name = model_name
            self._limits = _MODEL_LIMITS.get(model_name, _DEFAULT_LIMITS)
            logger.info(
                f"[llm_client] ✓ Switched model: {old_model} → {model_name} "
                f"(budget: {self._limits['tpd']:,} TPD / {self._limits['tpm']:,} TPM)"
            )
            self._flush_model_status()
            return True
        except Exception as e:
            logger.error(f"[llm_client] Failed to switch to {model_name}: {e}")
            self._exhausted_models.add(model_name)
            return False

    def _switch_to_next_model(self) -> bool:
        """
        Switch self._llm to the next non-exhausted model in _FALLBACK_ORDER.
        Returns True if a switch was made, False if all models are exhausted.
        """
        for model_name in _FALLBACK_ORDER:
            if model_name not in self._exhausted_models:
                if self._switch_to_model(model_name):
                    return True
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

    def _flush_model_status(self) -> None:
        """
        Write data/model_status.json so the dashboard can show per-model availability.

        Schema:
          active      — model currently being used
          all_models  — ordered fallback list
          tpd_exhausted — models that hit their daily token limit
          unavailable   — models that returned 404 / decommissioned
          last_updated  — ISO timestamp
        """
        try:
            data = {
                "active":        self.config.model_name,
                "all_models":    _FALLBACK_ORDER,
                "tpd_exhausted": sorted(self._tpd_exhausted),
                "unavailable":   sorted(self._unavailable),
                "last_updated":  datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            self._status_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug(f"[llm_client] model_status flush error (non-fatal): {exc}")
