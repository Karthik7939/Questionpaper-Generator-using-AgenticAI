"""
agents/question_generator_agent.py

Question Generator Agent for the Agentic Question Paper Generator.

Responsibilities:
  - Read syllabus_topics and question_distribution from AgentState
  - Call Groq LLM via LLMService to generate exam questions
  - Validate every generated QuestionItem for required fields and valid values
  - Write List[QuestionItem] to AgentState.generated_questions
  - Handle errors gracefully by appending to AgentState.errors
"""

from typing import Any

from app.models.state import AgentState, QuestionDistribution, QuestionItem
from app.prompts.question_prompt import (
    QUESTION_SYSTEM_PROMPT,
    build_question_user_prompt,
)
from app.services.llm_service import LLMService
from app.services.logger import log_execution_time, setup_logger

logger = setup_logger(__name__)

AGENT_NAME = "QuestionGeneratorAgent"

# Valid constraint sets — used for field-level validation
_VALID_MARKS = {2, 5, 10, 15}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_QUESTION_TYPES = {"short", "brief", "long", "essay"}

# Maps marks value to expected question_type (for auto-correction)
_MARKS_TO_TYPE: dict[int, str] = {
    2: "short",
    5: "brief",
    10: "long",
    15: "essay",
}


class QuestionGeneratorAgent:
    """
    Generates university-level exam questions from syllabus topics.

    LangGraph node function: question_generator_agent_node()
    """

    def __init__(self) -> None:
        self.llm = LLMService()
        logger.info(f"{AGENT_NAME} initialized.")

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        Reads:  state["syllabus_topics"], state["question_distribution"]
        Writes: state["generated_questions"], state["current_agent"],
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
        # 1. Guard: workflow must not already be in failed state
        # ------------------------------------------------------------------
        if state.get("status") == "failed":
            logger.warning(
                f"{AGENT_NAME}: Skipping because workflow status is 'failed'."
            )
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 2. Guard: syllabus_topics must be non-empty
        # ------------------------------------------------------------------
        syllabus_topics: list = state.get("syllabus_topics", [])
        if not syllabus_topics:
            error_msg = (
                f"{AGENT_NAME}: syllabus_topics is empty. "
                "Cannot generate questions without extracted syllabus."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 3. Guard: question_distribution must be set
        # ------------------------------------------------------------------
        distribution: QuestionDistribution | None = state.get("question_distribution")
        if not distribution:
            error_msg = (
                f"{AGENT_NAME}: question_distribution is not set in state. "
                "Cannot generate questions without knowing the required counts."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 4. Extract distribution values (all typed fields from QuestionDistribution)
        # ------------------------------------------------------------------
        total_marks: int = distribution["total_marks"]
        two_mark_count: int = distribution["two_mark_questions"]
        five_mark_count: int = distribution["five_mark_questions"]
        ten_mark_count: int = distribution["ten_mark_questions"]
        fifteen_mark_count: int = distribution["fifteen_mark_questions"]
        easy_pct: int = distribution["easy_percentage"]
        medium_pct: int = distribution["medium_percentage"]
        hard_pct: int = distribution["hard_percentage"]

        logger.info(
            f"{AGENT_NAME}: Generating questions — "
            f"total_marks={total_marks}, "
            f"2M×{two_mark_count}, 5M×{five_mark_count}, "
            f"10M×{ten_mark_count}, 15M×{fifteen_mark_count}"
        )

        # ------------------------------------------------------------------
        # 5. Call LLM to generate questions
        # ------------------------------------------------------------------
        try:
            content_context: str = state.get("content_context") or ""

            user_prompt = build_question_user_prompt(
                syllabus_topics=syllabus_topics,
                content_context=content_context,
                total_marks=total_marks,
                two_mark_count=two_mark_count,
                five_mark_count=five_mark_count,
                ten_mark_count=ten_mark_count,
                fifteen_mark_count=fifteen_mark_count,
                easy_pct=easy_pct,
                medium_pct=medium_pct,
                hard_pct=hard_pct,
            )
            raw_data: Any = self.llm.call_llm_for_json(
                system_prompt=QUESTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name=AGENT_NAME,
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = f"{AGENT_NAME}: LLM call failed — {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 6. Validate response is a non-empty list
        # ------------------------------------------------------------------
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            error_msg = (
                f"{AGENT_NAME}: LLM returned an unexpected format. "
                f"Expected a non-empty JSON array, got: {type(raw_data).__name__}"
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 7. Validate and coerce each item into QuestionItem structure
        # ------------------------------------------------------------------
        generated_questions: list[QuestionItem] = []
        item_warnings: list[str] = []

        for idx, item in enumerate(raw_data):
            if not isinstance(item, dict):
                item_warnings.append(
                    f"{AGENT_NAME}: Item at index {idx} is not a dict — skipped."
                )
                continue

            q_id = item.get("id")
            unit = item.get("unit")
            topic = item.get("topic")
            question = item.get("question")
            marks = item.get("marks")
            difficulty = item.get("difficulty")
            question_type = item.get("question_type")

            # --- Required string fields ---
            if not q_id or not isinstance(q_id, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} missing 'id' — skipped."
                )
                continue
            if not unit or not isinstance(unit, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) missing 'unit' — skipped."
                )
                continue
            if not topic or not isinstance(topic, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) missing 'topic' — skipped."
                )
                continue
            if not question or not isinstance(question, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) missing 'question' — skipped."
                )
                continue

            # --- Marks: coerce to int, validate against allowed values ---
            try:
                marks = int(marks)
            except (ValueError, TypeError):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) invalid 'marks' value '{marks}' — skipped."
                )
                continue
            if marks not in _VALID_MARKS:
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) 'marks'={marks} not in {_VALID_MARKS} — skipped."
                )
                continue

            # --- Difficulty: validate and lowercase ---
            if not difficulty or not isinstance(difficulty, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) missing 'difficulty' — skipped."
                )
                continue
            difficulty = difficulty.strip().lower()
            if difficulty not in _VALID_DIFFICULTIES:
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) invalid difficulty '{difficulty}' — skipped."
                )
                continue

            # --- question_type: auto-correct from marks if missing/invalid ---
            if not question_type or question_type.strip().lower() not in _VALID_QUESTION_TYPES:
                corrected = _MARKS_TO_TYPE[marks]
                item_warnings.append(
                    f"{AGENT_NAME}: Item {idx} (id={q_id}) invalid question_type "
                    f"'{question_type}' — auto-corrected to '{corrected}'."
                )
                question_type = corrected
            else:
                question_type = question_type.strip().lower()

            generated_questions.append(
                QuestionItem(
                    id=q_id.strip(),
                    unit=unit.strip(),
                    topic=topic.strip(),
                    question=question.strip(),
                    marks=marks,
                    difficulty=difficulty,
                    question_type=question_type,
                )
            )

        # Log per-item warnings (non-fatal)
        for warning in item_warnings:
            logger.warning(warning)
        errors.extend(item_warnings)

        # ------------------------------------------------------------------
        # 8. Final guard: at least one valid question must have been generated
        # ------------------------------------------------------------------
        if not generated_questions:
            error_msg = (
                f"{AGENT_NAME}: No valid questions could be extracted from "
                "the LLM response. Check prompt or syllabus content."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "generated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: Successfully generated {len(generated_questions)} question(s)."
        )

        # ------------------------------------------------------------------
        # 9. Return partial state update
        # ------------------------------------------------------------------
        return {
            "generated_questions": generated_questions,
            "current_agent": AGENT_NAME,
            "status": "running",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
_question_generator_agent = QuestionGeneratorAgent()


def question_generator_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph-compatible node function for the Question Generator Agent.

    This is the function registered with StateGraph.add_node().

    Args:
        state: The shared AgentState dict passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    return _question_generator_agent(state)
