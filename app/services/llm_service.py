"""
services/llm_service.py

Central LLM service for the Agentic Question Paper Generator.
All agent calls to the Groq LLM go through this single service.

Responsibilities:
  - Initialize and manage the ChatGroq client
  - Retry failed API calls with exponential backoff
  - Extract and log token usage
  - Parse JSON responses safely
  - Provide a clean call_llm() interface for all agents
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.services.logger import log_token_usage, setup_logger

logger = setup_logger(__name__)


class LLMService:
    """
    Singleton-style service that wraps ChatGroq.

    Usage:
        llm_service = LLMService()
        response = llm_service.call_llm(system_prompt, user_prompt)
        data = llm_service.call_llm_for_json(system_prompt, user_prompt)
    """

    def __init__(self) -> None:
        self._model: ChatGroq | None = None
        logger.info(
            f"LLMService initialized — model: {settings.llm.MODEL_NAME} | "
            f"temperature: {settings.llm.TEMPERATURE}"
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_model(self) -> ChatGroq:
        """Create and return a configured ChatGroq instance."""
        api_key = (settings.llm.GROQ_API_KEY or "").strip()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Please add it to your .env file or set it in the environment."
            )

        try:
            model = ChatGroq(
                model=settings.llm.MODEL_NAME,
                groq_api_key=api_key,
                temperature=settings.llm.TEMPERATURE,
            )
            return model
        except Exception as exc:
            logger.error(f"Failed to initialize ChatGroq: {exc}")
            raise

    def _get_model(self) -> ChatGroq:
        """Lazily initialize the Groq client on first use."""
        if self._model is None:
            self._model = self._initialize_model()
        return self._model

    # ------------------------------------------------------------------
    # Core LLM call with retry
    # ------------------------------------------------------------------

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "UnknownAgent",
    ) -> str:
        """
        Send a prompt to Groq and return the response text.
        Retries on failure with exponential backoff.

        Args:
            system_prompt: The system/instruction prompt for the LLM.
            user_prompt:   The user/content prompt (the actual task).
            agent_name:    Name of the calling agent (used for logging).

        Returns:
            The LLM's response as a plain string.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            model = self._get_model()
        except EnvironmentError as exc:
            raise RuntimeError(f"[{agent_name}] {exc}") from exc

        last_exception: Exception | None = None

        for attempt in range(1, settings.llm.MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[{agent_name}] Calling Groq API "
                    f"(attempt {attempt}/{settings.llm.MAX_RETRIES}) ..."
                )
                response = model.invoke(messages)

                # Extract and log token usage if available
                self._log_usage(response, agent_name)

                content = response.content
                if not content or not content.strip():
                    raise ValueError("LLM returned an empty response.")

                logger.info(f"[{agent_name}] Groq API call succeeded.")
                return content.strip()

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"[{agent_name}] Attempt {attempt} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < settings.llm.MAX_RETRIES:
                    wait = settings.llm.RETRY_DELAY_SECONDS * attempt  # Exponential backoff
                    logger.info(f"[{agent_name}] Retrying in {wait:.1f}s ...")
                    time.sleep(wait)

        raise RuntimeError(
            f"[{agent_name}] All {settings.llm.MAX_RETRIES} Groq API attempts failed. "
            f"Last error: {last_exception}"
        )

    # ------------------------------------------------------------------
    # JSON-specific call
    # ------------------------------------------------------------------

    def call_llm_for_json(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "UnknownAgent",
    ) -> Any:
        """
        Call the LLM and parse the response as JSON.

        The system prompt should instruct the LLM to return valid JSON.
        This method strips markdown code fences if the model wraps its
        response in ```json ... ``` blocks.

        Args:
            system_prompt: Prompt instructing JSON output.
            user_prompt:   The task prompt.
            agent_name:    Name of the calling agent (used for logging).

        Returns:
            Parsed Python object (dict or list).

        Raises:
            ValueError:   If the response cannot be parsed as valid JSON.
            RuntimeError: If all retry attempts are exhausted.
        """
        raw_response = self.call_llm(system_prompt, user_prompt, agent_name)

        try:
            cleaned = self._strip_code_fences(raw_response)
            parsed = json.loads(cleaned)
            logger.info(f"[{agent_name}] JSON response parsed successfully.")
            return parsed

        except json.JSONDecodeError as exc:
            logger.error(
                f"[{agent_name}] Failed to parse JSON response.\n"
                f"Raw response:\n{raw_response}\n"
                f"Error: {exc}"
            )
            raise ValueError(
                f"[{agent_name}] LLM returned invalid JSON: {exc}\n"
                f"Response was:\n{raw_response}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_usage(self, response: Any, agent_name: str) -> None:
        """Extract token usage metadata from the LLM response if available."""
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                log_token_usage(
                    logger=logger,
                    agent_name=agent_name,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                )
        except Exception:
            pass  # Token logging is best-effort, never crash on it

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        Remove markdown code fences from LLM responses.

        Handles:
            ```json ... ```
            ``` ... ```
        """
        text = text.strip()

        # Strip ```json ... ``` or ``` ... ``` wrappers
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].strip().startswith("```"):
                lines = lines[1:]
            text = "\n".join(lines).strip()

        return text
