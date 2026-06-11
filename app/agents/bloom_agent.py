"""
agents/bloom_agent.py

Bloom Taxonomy Agent for the Agentic Question Paper Generator.

Responsibilities:
  - Read generated_questions from AgentState
  - Call Groq LLM via LLMService to classify each question by Bloom's Taxonomy level
  - Validate every returned BloomItem for required fields and valid level values
  - Ensure every input question has a corresponding output BloomItem
  - Write List[BloomItem] to AgentState.bloom_analysis
  - Handle errors gracefully by appending to AgentState.errors
"""

from typing import Any

from app.models.state import AgentState, BloomItem
from app.prompts.bloom_prompt import (
    BLOOM_SYSTEM_PROMPT,
    build_bloom_user_prompt,
)
from app.services.llm_service import LLMService
from app.services.logger import log_execution_time, setup_logger

logger = setup_logger(__name__)

AGENT_NAME = "BloomAgent"

# Valid Bloom levels exactly as defined in state.py and the prompt
_VALID_BLOOM_LEVELS = {
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
}

# Title-case normalization map — handles LLM capitalisation variants
_BLOOM_NORMALISE: dict[str, str] = {
    level.lower(): level for level in _VALID_BLOOM_LEVELS
}


class BloomAgent:
    """
    Classifies generated exam questions into Bloom's Taxonomy levels.

    LangGraph node function: bloom_agent_node()
    """

    def __init__(self) -> None:
        self.llm = LLMService()
        logger.info(f"{AGENT_NAME} initialized.")

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        Reads:  state["generated_questions"]
        Writes: state["bloom_analysis"], state["current_agent"],
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
        # 1. Guard: propagate failed state without executing
        # ------------------------------------------------------------------
        if state.get("status") == "failed":
            logger.warning(
                f"{AGENT_NAME}: Skipping because workflow status is 'failed'."
            )
            return {
                "bloom_analysis": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 2. Guard: generated_questions must be non-empty
        # ------------------------------------------------------------------
        generated_questions: list = state.get("generated_questions", [])
        if not generated_questions:
            error_msg = (
                f"{AGENT_NAME}: generated_questions is empty. "
                "Cannot classify questions without generated content."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "bloom_analysis": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Classifying {len(generated_questions)} question(s) "
            "by Bloom's Taxonomy level."
        )

        # ------------------------------------------------------------------
        # 3. Call LLM to classify questions
        # ------------------------------------------------------------------
        try:
            user_prompt = build_bloom_user_prompt(generated_questions)
            raw_data: Any = self.llm.call_llm_for_json(
                system_prompt=BLOOM_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name=AGENT_NAME,
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = f"{AGENT_NAME}: LLM call failed — {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "bloom_analysis": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 4. Validate response is a non-empty list
        # ------------------------------------------------------------------
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            error_msg = (
                f"{AGENT_NAME}: LLM returned an unexpected format. "
                f"Expected a non-empty JSON array, got: {type(raw_data).__name__}"
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "bloom_analysis": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 5. Build a lookup of input questions by id for cross-reference
        # ------------------------------------------------------------------
        input_by_id: dict[str, dict] = {
            q["id"]: q for q in generated_questions if "id" in q
        }

        # ------------------------------------------------------------------
        # 6. Validate and coerce each item into BloomItem structure
        # ------------------------------------------------------------------
        bloom_analysis: list[BloomItem] = []
        item_warnings: list[str] = []
        seen_ids: set[str] = set()

        for idx, item in enumerate(raw_data):
            if not isinstance(item, dict):
                item_warnings.append(
                    f"{AGENT_NAME}: Item at index {idx} is not a dict — skipped."
                )
                continue

            q_id = item.get("id")
            question = item.get("question")
            marks = item.get("marks")
            difficulty = item.get("difficulty")
            bloom_level = item.get("bloom_level")
            bloom_justification = item.get("bloom_justification")

            # --- id ---
            if not q_id or not isinstance(q_id, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item at index {idx} missing 'id' — skipped."
                )
                continue

            # --- deduplicate ---
            if q_id in seen_ids:
                item_warnings.append(
                    f"{AGENT_NAME}: Duplicate id '{q_id}' in Bloom response — skipped."
                )
                continue
            seen_ids.add(q_id)

            # --- question text: fall back to input question if LLM changed it ---
            if not question or not isinstance(question, str):
                if q_id in input_by_id:
                    question = input_by_id[q_id]["question"]
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'question' — "
                        "restored from generated_questions."
                    )
                else:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'question' and "
                        "not found in input — skipped."
                    )
                    continue

            # --- marks: coerce to int, fall back to input ---
            try:
                marks = int(marks)
            except (ValueError, TypeError):
                if q_id in input_by_id:
                    marks = input_by_id[q_id]["marks"]
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' — "
                        "restored from generated_questions."
                    )
                else:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' and "
                        "not found in input — skipped."
                    )
                    continue

            # --- difficulty: fallback to input ---
            if not difficulty or not isinstance(difficulty, str):
                if q_id in input_by_id:
                    difficulty = input_by_id[q_id]["difficulty"]
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'difficulty' — "
                        "restored from generated_questions."
                    )
                else:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'difficulty' and "
                        "not found in input — skipped."
                    )
                    continue
            difficulty = difficulty.strip().lower()

            # --- bloom_level: normalise capitalisation, validate ---
            if not bloom_level or not isinstance(bloom_level, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'bloom_level' — skipped."
                )
                continue

            normalised_level = _BLOOM_NORMALISE.get(bloom_level.strip().lower())
            if not normalised_level:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' has unrecognised bloom_level "
                    f"'{bloom_level}'. Must be one of {sorted(_VALID_BLOOM_LEVELS)} — skipped."
                )
                continue

            # --- bloom_justification: use a default if empty ---
            if not bloom_justification or not isinstance(bloom_justification, str):
                bloom_justification = (
                    f"Classified as {normalised_level} based on the cognitive demand of the question."
                )
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'bloom_justification' — "
                    "default justification applied."
                )

            bloom_analysis.append(
                BloomItem(
                    id=q_id.strip(),
                    question=question.strip(),
                    marks=marks,
                    difficulty=difficulty,
                    bloom_level=normalised_level,
                    bloom_justification=bloom_justification.strip(),
                )
            )

        # Log per-item warnings (non-fatal)
        for warning in item_warnings:
            logger.warning(warning)
        errors.extend(item_warnings)

        # ------------------------------------------------------------------
        # 7. Warn about any input questions that received no Bloom classification
        # ------------------------------------------------------------------
        classified_ids = {item["id"] for item in bloom_analysis}
        for q in generated_questions:
            if q.get("id") and q["id"] not in classified_ids:
                errors.append(
                    f"{AGENT_NAME}: Question '{q['id']}' was not classified "
                    "by the LLM and has been omitted from bloom_analysis."
                )

        # ------------------------------------------------------------------
        # 8. Final guard: at least one classification must exist
        # ------------------------------------------------------------------
        if not bloom_analysis:
            error_msg = (
                f"{AGENT_NAME}: No valid Bloom classifications could be extracted "
                "from the LLM response."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "bloom_analysis": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Successfully classified {len(bloom_analysis)} question(s). "
            f"Levels used: {sorted({item['bloom_level'] for item in bloom_analysis})}"
        )

        # ------------------------------------------------------------------
        # 9. Return partial state update
        # ------------------------------------------------------------------
        return {
            "bloom_analysis": bloom_analysis,
            "current_agent": AGENT_NAME,
            "status": "running",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
_bloom_agent = BloomAgent()


def bloom_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph-compatible node function for the Bloom Taxonomy Agent.

    This is the function registered with StateGraph.add_node().

    Args:
        state: The shared AgentState dict passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    return _bloom_agent(state)
