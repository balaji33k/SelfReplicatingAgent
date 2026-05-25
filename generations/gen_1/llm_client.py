"""
llm_client.py — Multi-provider LLM client with automatic failover.

Provider / model chain (auto-detected, tried in order):
  1. Groq primary   — GROQ_API_KEY + llama-3.3-70b-versatile  (100k TPD, 70B params)
  2. Groq fallback  — GROQ_API_KEY + mixtral-8x7b-32768        (500k TPD, 46.7B MoE)
  3. Gemini         — GEMINI_API_KEY / GOOGLE_API_KEY          (1.5M TPD free)
  4. Vertex AI      — GOOGLE_GENAI_USE_VERTEXAI=TRUE + GCLOUD_ACCESS_TOKEN

On HTTP 429 from any provider/model, switches to the next entry immediately
(no wasted sleep). On other transient errors, retries with exponential backoff.

Env-var overrides:
  GROQ_FALLBACK_MODEL    — replaces llama-3.1-8b-instant (set to "" to disable)
  GEMINI_FALLBACK_MODEL  — replaces gemini-2.0-flash
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_RETRIES = 7
RETRY_BASE_WAIT = 5

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_PRIMARY_MODEL    = "llama-3.3-70b-versatile"  # 100k TPD, 70B params
DEFAULT_GROQ_FALLBACK_MODEL   = "mixtral-8x7b-32768"        # 500k TPD, 46.7B params MoE
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"


class LLMClient:
    """
    Universal LLM client with automatic failover across providers and models.
    On Groq 429: tries Groq fallback model first, then Gemini, then Vertex.
    No code changes needed — configure via environment variables.
    """

    def __init__(self, config):
        self.config = config  # LLMConfig — model_name, temperature, timeout_seconds

    # ------------------------------------------------------------------
    # Provider chain
    # ------------------------------------------------------------------

    def _get_provider_chain(self) -> List[Tuple[str, str, str]]:
        """
        Return ordered list of (provider_type, credential, model) triples.
        Each entry is tried in sequence; 429 → next entry.

        provider_type: 'groq' | 'gemini' | 'vertex'
        """
        chain: List[Tuple[str, str, str]] = []

        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            # 1. Primary Groq model (100k TPD)
            primary = self._groq_model()
            chain.append(("groq", groq_key, primary))

            # 2. Fallback Groq model (500k TPD) — disable by setting GROQ_FALLBACK_MODEL=""
            fallback_model = os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL).strip()
            if fallback_model and fallback_model != primary:
                chain.append(("groq", groq_key, fallback_model))

        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
            token = os.getenv("GCLOUD_ACCESS_TOKEN", "").strip()
            if token:
                chain.append(("vertex", token, self.config.model_name))
        else:
            gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if gemini_key:
                gemini_model = self._gemini_model()
                chain.append(("gemini", gemini_key, gemini_model))

        if not chain:
            raise EnvironmentError(
                "No API key found. Set GROQ_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY."
            )
        return chain

    # ------------------------------------------------------------------
    # Model name helpers
    # ------------------------------------------------------------------

    def _groq_model(self) -> str:
        """Return primary Groq model — swap Gemini names if config was set for Gemini."""
        model = self.config.model_name
        if "gemini" in model.lower():
            model = DEFAULT_GROQ_PRIMARY_MODEL
            logger.info(f"[llm_client] Gemini model name in config; using {model} for Groq")
        return model

    def _gemini_model(self) -> str:
        """Return Gemini model name — use configured name if it is a Gemini model."""
        model = self.config.model_name
        if "gemini" in model.lower():
            return model
        return os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL)

    # ------------------------------------------------------------------
    # Request builders
    # ------------------------------------------------------------------

    def _build_groq_request(self, prompt: str, api_key: str, model: str) -> urllib.request.Request:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": 8192,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "python-groq/0.9.0",
        }
        return urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
        )

    def _build_gemini_request(self, prompt: str, api_key: str, model: str) -> urllib.request.Request:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )

    def _build_vertex_request(self, prompt: str, token: str, model: str) -> urllib.request.Request:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "codingagentproject-487605")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        return urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )

    def _build_request(
        self, provider: str, credential: str, model: str, prompt: str
    ) -> urllib.request.Request:
        if provider == "groq":
            return self._build_groq_request(prompt, credential, model)
        elif provider == "vertex":
            return self._build_vertex_request(prompt, credential, model)
        else:
            return self._build_gemini_request(prompt, credential, model)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict, provider: str) -> str:
        if provider == "groq":
            return data["choices"][0]["message"]["content"]
        else:  # gemini / vertex
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def call(self, prompt: str) -> str:
        """
        Send a prompt and return the response text.

        Failover strategy:
          • HTTP 429 from any entry → switch to next in chain immediately (no sleep).
          • Other HTTP errors or network errors → retry current entry with
            exponential backoff up to MAX_RETRIES.
          • All entries exhausted → raise RuntimeError with full details.
        """
        chain = self._get_provider_chain()
        provider_idx = 0
        last_error = ""
        attempt = 0

        chain_labels = [f"{p}:{m}" for p, _, m in chain]
        logger.debug(f"[llm_client] Chain: {chain_labels}")

        while attempt < MAX_RETRIES and provider_idx < len(chain):
            provider, credential, model = chain[provider_idx]

            try:
                req = self._build_request(provider, credential, model, prompt)
                logger.debug(f"[llm_client] Attempt {attempt+1} — {provider}:{model}")
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode())
                    return self._parse_response(data, provider)

            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last_error = f"HTTP {e.code} {e.reason} — {body[:300]}"

                if e.code == 429:
                    next_idx = provider_idx + 1
                    if next_idx < len(chain):
                        next_label = f"{chain[next_idx][0]}:{chain[next_idx][2]}"
                        logger.warning(
                            f"[llm_client] {provider}:{model} rate-limited (429). "
                            f"Switching to {next_label} immediately."
                        )
                        provider_idx = next_idx
                        # Don't count this as a retry attempt — just switch
                        continue
                    else:
                        # No more fallbacks — wait and retry last entry
                        wait = RETRY_BASE_WAIT * (2 ** attempt)
                        logger.error(
                            f"[llm_client] {provider}:{model} rate-limited (429), no more "
                            f"fallbacks. Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})..."
                        )
                        attempt += 1
                        if attempt < MAX_RETRIES:
                            time.sleep(wait)
                else:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)
                    logger.error(
                        f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}) "
                        f"via {provider}:{model}: {last_error}. Retrying in {wait}s..."
                    )
                    attempt += 1
                    if attempt < MAX_RETRIES:
                        time.sleep(wait)

            except Exception as e:
                last_error = str(e)
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                logger.error(
                    f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}) "
                    f"via {provider}:{model}: {last_error}. Retrying in {wait}s..."
                )
                attempt += 1
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        tried = chain_labels[:provider_idx + 1]
        raise RuntimeError(
            f"LLM call failed after {attempt} attempts across providers {tried}. "
            f"Last error: {last_error}"
        )

    def chat_completion(
        self, messages: List[Dict], response_format: Optional[Dict] = None
    ) -> str:
        """
        Chat-style interface — flattens messages to a single prompt.
        Kept for backwards compatibility. `response_format` is ignored.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[Instructions]\n{content}")
            else:
                parts.append(content)
        return self.call("\n\n".join(parts))
