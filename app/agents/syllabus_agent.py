"""
agents/syllabus_agent.py

Syllabus Agent for the Agentic Question Paper Generator.

Responsibilities:
  - Read RAG-retrieved syllabus context from AgentState.syllabus_context
  - Call Groq LLM via LLMService to extract structured syllabus topics
  - Validate the JSON response structure
  - Write List[SyllabusTopic] to AgentState.syllabus_topics
  - Handle errors gracefully by appending to AgentState.errors
"""

from typing import Any

from app.models.state import AgentState, SyllabusTopic
from app.prompts.syllabus_prompt import (
    SYLLABUS_SYSTEM_PROMPT,
    build_syllabus_user_prompt,
)
from app.services.llm_service import LLMService
from app.services.logger import log_execution_time, setup_logger

logger = setup_logger(__name__)

AGENT_NAME = "SyllabusAgent"


class SyllabusAgent:
    """
    Extracts structured syllabus topics from RAG-retrieved document chunks.

    LangGraph node function: syllabus_agent_node()
    This class is instantiated once; its __call__ method is the node function.
    """

    def __init__(self) -> None:
        self.llm = LLMService()
        logger.info(f"{AGENT_NAME} initialized.")

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        Reads: state["syllabus_context"]
        Writes: state["syllabus_topics"], state["current_agent"],
                state["status"], state["errors"]

        Args:
            state: The shared AgentState dict.

        Returns:
            Partial state update dict for LangGraph to merge.
        """
        with log_execution_time(logger, AGENT_NAME):
            return self._run(state)

    def _run(self, state: AgentState) -> dict[str, Any]:
        """Core logic separated from the context manager for clarity."""

        errors: list[str] = list(state.get("errors", []))

        # ------------------------------------------------------------------
        # 1. Guard: syllabus_context must exist and be non-empty
        # ------------------------------------------------------------------
        syllabus_context: str | None = state.get("syllabus_context")

        if not syllabus_context or not syllabus_context.strip():
            error_msg = (
                f"{AGENT_NAME}: syllabus_context is empty or missing. "
                "Cannot extract syllabus topics from RAG chunks."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "syllabus_topics": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 2. Call LLM to extract syllabus topics
        # ------------------------------------------------------------------
        try:
            user_prompt = build_syllabus_user_prompt(syllabus_context)
            raw_data: Any = self.llm.call_llm_for_json(
                system_prompt=SYLLABUS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name=AGENT_NAME,
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = f"{AGENT_NAME}: LLM call failed — {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "syllabus_topics": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 3. Validate response is a non-empty list
        # ------------------------------------------------------------------
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            error_msg = (
                f"{AGENT_NAME}: LLM returned an unexpected format. "
                f"Expected a non-empty JSON array, got: {type(raw_data).__name__}"
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "syllabus_topics": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 4. Validate and coerce each item into SyllabusTopic structure
        # ------------------------------------------------------------------
        syllabus_topics: list[SyllabusTopic] = []
        validation_errors: list[str] = []

        for idx, item in enumerate(raw_data):
            if not isinstance(item, dict):
                validation_errors.append(
                    f"{AGENT_NAME}: Item at index {idx} is not a dict — skipped."
                )
                continue

            unit_number = item.get("unit_number")
            unit_name = item.get("unit_name")
            topics = item.get("topics")

            # Check required fields
            if unit_number is None:
                validation_errors.append(
                    f"{AGENT_NAME}: Item {idx} missing 'unit_number' — skipped."
                )
                continue
            if not unit_name or not isinstance(unit_name, str):
                validation_errors.append(
                    f"{AGENT_NAME}: Item {idx} missing or invalid 'unit_name' — skipped."
                )
                continue
            if not isinstance(topics, list) or len(topics) == 0:
                validation_errors.append(
                    f"{AGENT_NAME}: Item {idx} ('{unit_name}') has no topics — skipped."
                )
                continue

            # Coerce unit_number to int in case LLM returns it as a string
            try:
                unit_number = int(unit_number)
            except (ValueError, TypeError):
                validation_errors.append(
                    f"{AGENT_NAME}: Item {idx} 'unit_number' is not a valid integer — skipped."
                )
                continue

            # Filter out non-string topics
            clean_topics: list[str] = [
                str(t).strip() for t in topics if str(t).strip()
            ]
            if not clean_topics:
                validation_errors.append(
                    f"{AGENT_NAME}: Item {idx} ('{unit_name}') has no valid topic strings — skipped."
                )
                continue

            syllabus_topics.append(
                SyllabusTopic(
                    unit_number=unit_number,
                    unit_name=unit_name.strip(),
                    topics=clean_topics,
                )
            )

        # Log any per-item validation warnings (non-fatal)
        for warning in validation_errors:
            logger.warning(warning)
        errors.extend(validation_errors)

        # ------------------------------------------------------------------
        # 5. Final guard: at least one valid unit must have been extracted
        # ------------------------------------------------------------------
        if not syllabus_topics:
            error_msg = (
                f"{AGENT_NAME}: No valid syllabus units could be extracted "
                "from the LLM response. Check the uploaded document format."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "syllabus_topics": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Successfully extracted {len(syllabus_topics)} unit(s) "
            f"with {sum(len(u['topics']) for u in syllabus_topics)} total topic(s)."
        )

        # ------------------------------------------------------------------
        # 6. Return partial state update
        # ------------------------------------------------------------------
        return {
            "syllabus_topics": syllabus_topics,
            "current_agent": AGENT_NAME,
            "status": "running",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
# Instantiated once at import time; reused across workflow invocations.
_syllabus_agent = SyllabusAgent()


def syllabus_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph-compatible node function for the Syllabus Agent.

    This is the function registered with StateGraph.add_node().
    It delegates to the SyllabusAgent singleton.

    Args:
        state: The shared AgentState dict passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    return _syllabus_agent(state)
