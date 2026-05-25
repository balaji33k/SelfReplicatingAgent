"""
llm_client.py — LangChain/Groq-backed LLM client.

Uses ChatGroq from langchain-groq for all LLM calls.
Exposes the same .call() / .chat_completion() interface that every agent
in agent_base.py uses — no agent files need to change.

Provider: Groq (GROQ_API_KEY).
Model: set in config.py (default: meta-llama/llama-4-scout-17b-16e-instruct).
Retries: handled automatically by LangChain (max_retries in ChatGroq constructor).
"""
import logging
import os
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around ChatGroq that preserves the .call() / .chat_completion()
    interface used by all specialist agents via agent_base.SpecialistAgent.

    LangChain handles:
      - HTTP connection pooling
      - Automatic retries with exponential backoff (max_retries)
      - Token streaming (not used here, but available)
      - Rate-limit error handling
    """

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
        logger.info(
            f"[llm_client] ChatGroq initialised — "
            f"model={config.model_name} temperature={config.temperature}"
        )

    def call(self, prompt: str) -> str:
        """
        Send a single user prompt and return the response text.
        Called by every agent via SpecialistAgent._call_llm().
        """
        logger.debug(f"[llm_client] call() — prompt length={len(prompt)}")
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def chat_completion(
        self, messages: List[Dict], response_format: Optional[Dict] = None
    ) -> str:
        """
        Multi-turn chat interface.
        Kept for backwards compatibility. response_format is ignored.
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
        return response.content
