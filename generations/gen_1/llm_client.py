"""
llm_client.py — Multi-provider LLM client with automatic failover.

Provider priority (auto-detected via environment variables):
  1. Groq          — GROQ_API_KEY          (default if set)
  2. Gemini        — GEMINI_API_KEY / GOOGLE_API_KEY  (fallback)
  3. Vertex AI     — GOOGLE_GENAI_USE_VERTEXAI=TRUE + GCLOUD_ACCESS_TOKEN

On Groq HTTP 429 (rate limit), automatically switches to Gemini for that call.
On Gemini HTTP 429, waits with exponential backoff and retries.

Model selection:
  - Groq model: config.model_name (if it contains 'gemini', uses llama-3.3-70b-versatile)
  - Gemini model: config.model_name if it contains 'gemini', else GEMINI_FALLBACK_MODEL
    env var, else 'gemini-2.0-flash'
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

# Groq API endpoint (OpenAI-compatible)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Default Gemini fallback model when Groq is rate-limited
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"


class LLMClient:
    """
    Universal LLM client supporting Groq, Gemini, and Vertex AI.
    Provider is auto-detected from environment variables — no code changes needed.
    Automatically falls back to Gemini when Groq returns HTTP 429.
    """

    def __init__(self, config):
        self.config = config  # LLMConfig — model_name and temperature

    # ------------------------------------------------------------------
    # Provider discovery
    # ------------------------------------------------------------------

    def _get_provider_chain(self) -> List[Tuple[str, str]]:
        """
        Return ordered list of (provider, credential) pairs.
        Groq first (if key present), then Gemini/Vertex.
        """
        chain = []

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            chain.append(("groq", groq_key))

        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
            token = os.getenv("GCLOUD_ACCESS_TOKEN")
            if token:
                chain.append(("vertex", token))
        else:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                chain.append(("gemini", gemini_key))

        if not chain:
            raise EnvironmentError(
                "No API key found. Set GROQ_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY."
            )
        return chain

    # ------------------------------------------------------------------
    # Request builders
    # ------------------------------------------------------------------

    def _groq_model(self) -> str:
        model = self.config.model_name
        if "gemini" in model.lower():
            model = "llama-3.3-70b-versatile"
            logger.info(f"[llm_client] Gemini model name in config, using {model} for Groq")
        return model

    def _gemini_model(self) -> str:
        model = self.config.model_name
        if "gemini" in model.lower():
            return model
        # Configured for Groq but falling back to Gemini — pick sensible default
        fallback = os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL)
        return fallback

    def _build_groq_request(self, prompt: str, api_key: str) -> urllib.request.Request:
        payload = {
            "model": self._groq_model(),
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

    def _build_gemini_request(self, prompt: str, api_key: str) -> urllib.request.Request:
        model = self._gemini_model()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )

    def _build_vertex_request(self, prompt: str, token: str) -> urllib.request.Request:
        model = self.config.model_name
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

    def _build_request(self, provider: str, credential: str, prompt: str) -> urllib.request.Request:
        if provider == "groq":
            return self._build_groq_request(prompt, credential)
        elif provider == "vertex":
            return self._build_vertex_request(prompt, credential)
        else:
            return self._build_gemini_request(prompt, credential)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict, provider: str) -> str:
        if provider == "groq":
            return data["choices"][0]["message"]["content"]
        else:  # gemini / vertex
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ------------------------------------------------------------------
    # Main call with provider failover
    # ------------------------------------------------------------------

    def call(self, prompt: str) -> str:
        """
        Send a prompt and return the response text.

        Strategy:
          • Start with first provider in chain (usually Groq).
          • On HTTP 429 → if another provider is available, switch immediately
            (no sleep). This handles both TPM and TPD limits.
          • On other HTTP errors or network errors → retry current provider
            with exponential backoff.
          • After MAX_RETRIES transient failures, raise RuntimeError.
        """
        chain = self._get_provider_chain()
        provider_idx = 0
        last_error = ""
        attempt = 0

        logger.debug(
            f"[llm_client] Provider chain: {[p for p,_ in chain]} | "
            f"model={self.config.model_name}"
        )

        while attempt < MAX_RETRIES and provider_idx < len(chain):
            provider, credential = chain[provider_idx]

            try:
                req = self._build_request(provider, credential, prompt)
                logger.debug(f"[llm_client] Attempt {attempt+1} via {provider}")
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode())
                    return self._parse_response(data, provider)

            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last_error = f"HTTP {e.code} {e.reason} — {body[:300]}"

                if e.code == 429:
                    # Rate limit — check if we can failover
                    next_idx = provider_idx + 1
                    if next_idx < len(chain):
                        next_provider = chain[next_idx][0]
                        logger.warning(
                            f"[llm_client] {provider} rate-limited (429 TPD/TPM). "
                            f"Switching to {next_provider} immediately."
                        )
                        provider_idx = next_idx
                        # Don't increment `attempt` — the failover doesn't cost a retry
                        continue
                    else:
                        # No more providers; wait and retry same provider
                        wait = RETRY_BASE_WAIT * (2 ** attempt)
                        logger.error(
                            f"[llm_client] {provider} rate-limited (429), no fallback. "
                            f"Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}..."
                        )
                        attempt += 1
                        if attempt < MAX_RETRIES:
                            time.sleep(wait)
                else:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)
                    logger.error(
                        f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}) "
                        f"via {provider}: {last_error}. Retrying in {wait}s..."
                    )
                    attempt += 1
                    if attempt < MAX_RETRIES:
                        time.sleep(wait)

            except Exception as e:
                last_error = str(e)
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                logger.error(
                    f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}) "
                    f"via {provider}: {last_error}. Retrying in {wait}s..."
                )
                attempt += 1
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {attempt} attempts across providers "
            f"{[p for p,_ in chain[:provider_idx+1]]}. Last error: {last_error}"
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
