"""
agents/answerkey_agent.py

Answer Key Agent for the Agentic Question Paper Generator.

Responsibilities:
  - Read validated_questions from AgentState
  - Call Groq LLM via LLMService to generate model answers and marking schemes
  - Validate every returned AnswerKeyItem for required fields and types
  - Ensure every validated question receives a corresponding answer key entry
  - Write List[AnswerKeyItem] to AgentState.answer_key
  - Handle errors gracefully by appending to AgentState.errors
"""

from typing import Any

from app.models.state import AgentState, AnswerKeyItem
from app.prompts.answerkey_prompt import (
    ANSWERKEY_SYSTEM_PROMPT,
    build_answerkey_user_prompt,
)
from app.services.llm_service import LLMService
from app.services.logger import log_execution_time, setup_logger

logger = setup_logger(__name__)

AGENT_NAME = "AnswerKeyAgent"

# Valid marks — used to validate preserved marks in LLM response
_VALID_MARKS = {2, 5, 10, 15}


class AnswerKeyAgent:
    """
    Generates model answers and marking schemes for all validated questions.

    LangGraph node function: answerkey_agent_node()
    """

    def __init__(self) -> None:
        self.llm = LLMService()
        logger.info(f"{AGENT_NAME} initialized.")

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        Reads:  state["validated_questions"]
        Writes: state["answer_key"], state["current_agent"],
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
                "answer_key": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 2. Guard: validated_questions must be non-empty
        # ------------------------------------------------------------------
        validated_questions: list = state.get("validated_questions", [])
        if not validated_questions:
            error_msg = (
                f"{AGENT_NAME}: validated_questions is empty. "
                "Cannot generate answer key without validated questions."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "answer_key": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Generating answer key for "
            f"{len(validated_questions)} question(s)."
        )

        # ------------------------------------------------------------------
        # 3. Call LLM to generate model answers and marking schemes
        # ------------------------------------------------------------------
        try:
            content_context: str = state.get("content_context") or ""

            user_prompt = build_answerkey_user_prompt(
                validated_questions,
                content_context=content_context,
            )
            raw_data: Any = self.llm.call_llm_for_json(
                system_prompt=ANSWERKEY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name=AGENT_NAME,
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = f"{AGENT_NAME}: LLM call failed — {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "answer_key": [],
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
                "answer_key": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 5. Build lookup of validated questions by id for fallback
        # ------------------------------------------------------------------
        validated_by_id: dict[str, dict] = {
            q["id"]: q for q in validated_questions if "id" in q
        }

        # ------------------------------------------------------------------
        # 6. Validate and coerce each item into AnswerKeyItem structure
        # ------------------------------------------------------------------
        answer_key: list[AnswerKeyItem] = []
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
            model_answer = item.get("model_answer")
            key_points = item.get("key_points")
            marks_breakdown = item.get("marks_breakdown")

            # --- id ---
            if not q_id or not isinstance(q_id, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item at index {idx} missing 'id' — skipped."
                )
                continue

            q_id = q_id.strip()

            if q_id in seen_ids:
                item_warnings.append(
                    f"{AGENT_NAME}: Duplicate id '{q_id}' in answer key — skipped."
                )
                continue
            seen_ids.add(q_id)

            validated_src = validated_by_id.get(q_id, {})

            # --- question: fall back to validated_questions ---
            if not question or not isinstance(question, str):
                question = validated_src.get("question", "")
                if not question:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'question' "
                        "and no fallback found — skipped."
                    )
                    continue
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'question' — "
                    "restored from validated_questions."
                )

            # --- marks: coerce to int, fall back to validated source ---
            try:
                marks = int(marks)
            except (ValueError, TypeError):
                fallback_marks = validated_src.get("marks")
                if fallback_marks is None:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' "
                        "and no fallback found — skipped."
                    )
                    continue
                marks = int(fallback_marks)
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' — "
                    "restored from validated_questions."
                )
            if marks not in _VALID_MARKS:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' marks={marks} "
                    f"not in {_VALID_MARKS} — skipped."
                )
                continue

            # --- model_answer: must be a non-empty string ---
            if not model_answer or not isinstance(model_answer, str) or not model_answer.strip():
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing or empty 'model_answer' — skipped."
                )
                continue

            # --- key_points: must be a list of strings; filter empty entries ---
            if not isinstance(key_points, list) or len(key_points) == 0:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' 'key_points' is missing or empty — skipped."
                )
                continue
            clean_key_points: list[str] = [
                str(kp).strip() for kp in key_points if str(kp).strip()
            ]
            if not clean_key_points:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' 'key_points' contains "
                    "no valid strings — skipped."
                )
                continue

            # --- marks_breakdown: must be a non-empty string ---
            if not marks_breakdown or not isinstance(marks_breakdown, str) or not marks_breakdown.strip():
                # Construct a minimal fallback breakdown
                marks_breakdown = f"{marks} marks total"
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'marks_breakdown' — "
                    f"defaulted to '{marks_breakdown}'."
                )

            answer_key.append(
                AnswerKeyItem(
                    id=q_id,
                    question=question.strip(),
                    marks=marks,
                    model_answer=model_answer.strip(),
                    key_points=clean_key_points,
                    marks_breakdown=marks_breakdown.strip(),
                )
            )

        # Log per-item warnings (non-fatal)
        for warning in item_warnings:
            logger.warning(warning)
        errors.extend(item_warnings)

        # ------------------------------------------------------------------
        # 7. Warn about validated questions that received no answer key entry
        # ------------------------------------------------------------------
        answered_ids = {item["id"] for item in answer_key}
        for vq in validated_questions:
            if vq.get("id") and vq["id"] not in answered_ids:
                errors.append(
                    f"{AGENT_NAME}: Validated question '{vq['id']}' "
                    "has no answer key entry in the LLM response."
                )

        # ------------------------------------------------------------------
        # 8. Final guard: at least one answer key entry must exist
        # ------------------------------------------------------------------
        if not answer_key:
            error_msg = (
                f"{AGENT_NAME}: No valid answer key entries could be extracted "
                "from the LLM response. Check prompt or question content."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "answer_key": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Successfully generated {len(answer_key)} answer key entries."
        )

        # ------------------------------------------------------------------
        # 9. Return partial state update
        # ------------------------------------------------------------------
        return {
            "answer_key": answer_key,
            "current_agent": AGENT_NAME,
            "status": "running",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
_answerkey_agent = AnswerKeyAgent()


def answerkey_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph-compatible node function for the Answer Key Agent.

    This is the function registered with StateGraph.add_node().

    Args:
        state: The shared AgentState dict passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    return _answerkey_agent(state)
