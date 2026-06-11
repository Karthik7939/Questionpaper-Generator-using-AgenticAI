"""
agents/validation_agent.py

Validation Agent for the Agentic Question Paper Generator.

Responsibilities:
  - Read bloom_analysis, syllabus_topics, and question_distribution from AgentState
  - Call Groq LLM via LLMService to validate and correct the question set
  - Parse the LLM's validation report (is_valid, issues_found, validated_questions)
  - Validate each returned ValidatedQuestion for required fields and valid values
  - Write List[ValidatedQuestion] to AgentState.validated_questions
  - Handle errors gracefully by appending to AgentState.errors
"""

from typing import Any

from app.models.state import AgentState, QuestionDistribution, ValidatedQuestion
from app.prompts.validation_prompt import (
    VALIDATION_SYSTEM_PROMPT,
    build_validation_user_prompt,
)
from app.services.llm_service import LLMService
from app.services.logger import log_execution_time, setup_logger

logger = setup_logger(__name__)

AGENT_NAME = "ValidationAgent"

# Valid value sets for field-level validation
_VALID_MARKS = {2, 5, 10, 15}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_BLOOM_LEVELS = {
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
}
_BLOOM_NORMALISE: dict[str, str] = {
    level.lower(): level for level in _VALID_BLOOM_LEVELS
}
_VALID_QUESTION_TYPES = {"short", "brief", "long", "essay"}
_MARKS_TO_TYPE: dict[int, str] = {
    2: "short", 5: "brief", 10: "long", 15: "essay"
}


class ValidationAgent:
    """
    Validates the generated and Bloom-classified questions for exam readiness.

    LangGraph node function: validation_agent_node()
    """

    def __init__(self) -> None:
        self.llm = LLMService()
        logger.info(f"{AGENT_NAME} initialized.")

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        Reads:  state["bloom_analysis"], state["syllabus_topics"],
                state["question_distribution"]
        Writes: state["validated_questions"], state["current_agent"],
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
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 2. Guard: bloom_analysis must be non-empty
        # ------------------------------------------------------------------
        bloom_analysis: list = state.get("bloom_analysis", [])
        if not bloom_analysis:
            error_msg = (
                f"{AGENT_NAME}: bloom_analysis is empty. "
                "Cannot validate questions without Bloom-classified content."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 3. Guard: syllabus_topics must be non-empty
        # ------------------------------------------------------------------
        syllabus_topics: list = state.get("syllabus_topics", [])
        if not syllabus_topics:
            error_msg = (
                f"{AGENT_NAME}: syllabus_topics is empty. "
                "Cannot validate syllabus coverage without topic data."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 4. Extract distribution for validation constraints
        # ------------------------------------------------------------------
        distribution: QuestionDistribution | None = state.get("question_distribution")
        if not distribution:
            error_msg = (
                f"{AGENT_NAME}: question_distribution is not set. "
                "Cannot validate marks distribution without it."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        total_marks: int = distribution["total_marks"]
        two_mark_count: int = distribution["two_mark_questions"]
        five_mark_count: int = distribution["five_mark_questions"]
        ten_mark_count: int = distribution["ten_mark_questions"]
        fifteen_mark_count: int = distribution["fifteen_mark_questions"]

        logger.info(
            f"{AGENT_NAME}: Validating {len(bloom_analysis)} question(s) "
            f"against {len(syllabus_topics)} unit(s)."
        )

        # ------------------------------------------------------------------
        # 5. Call LLM to validate and correct the paper
        # ------------------------------------------------------------------
        try:
            content_context: str = state.get("content_context") or ""

            user_prompt = build_validation_user_prompt(
                bloom_analysis=bloom_analysis,
                syllabus_topics=syllabus_topics,
                content_context=content_context,
                total_marks=total_marks,
                two_mark_count=two_mark_count,
                five_mark_count=five_mark_count,
                ten_mark_count=ten_mark_count,
                fifteen_mark_count=fifteen_mark_count,
            )
            raw_data: Any = self.llm.call_llm_for_json(
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name=AGENT_NAME,
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = f"{AGENT_NAME}: LLM call failed — {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 6. Validate top-level response structure (must be a dict)
        # ------------------------------------------------------------------
        if not isinstance(raw_data, dict):
            error_msg = (
                f"{AGENT_NAME}: LLM returned an unexpected format. "
                f"Expected a JSON object, got: {type(raw_data).__name__}"
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 7. Extract and log validation report fields
        # ------------------------------------------------------------------
        is_valid: bool = bool(raw_data.get("is_valid", False))
        issues_found: list = raw_data.get("issues_found", [])
        validation_summary: str = raw_data.get("validation_summary", "")
        raw_validated_questions: Any = raw_data.get("validated_questions", [])

        if not is_valid and issues_found:
            logger.warning(
                f"{AGENT_NAME}: Validation found {len(issues_found)} issue(s):"
            )
            for issue in issues_found:
                logger.warning(f"  • {issue}")
                errors.append(f"{AGENT_NAME} issue: {issue}")
        else:
            logger.info(f"{AGENT_NAME}: Paper passed LLM validation.")

        if validation_summary:
            logger.info(f"{AGENT_NAME}: Summary — {validation_summary}")

        # ------------------------------------------------------------------
        # 8. Validate validated_questions is a non-empty list
        # ------------------------------------------------------------------
        if not isinstance(raw_validated_questions, list) or len(raw_validated_questions) == 0:
            error_msg = (
                f"{AGENT_NAME}: 'validated_questions' in LLM response is empty or missing."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        # ------------------------------------------------------------------
        # 9. Build Bloom lookup from bloom_analysis for field fallback
        # ------------------------------------------------------------------
        bloom_by_id: dict[str, dict] = {
            item["id"]: item for item in bloom_analysis if "id" in item
        }

        # ------------------------------------------------------------------
        # 10. Validate and coerce each item into ValidatedQuestion structure
        # ------------------------------------------------------------------
        validated_questions: list[ValidatedQuestion] = []
        item_warnings: list[str] = []
        seen_ids: set[str] = set()

        for idx, item in enumerate(raw_validated_questions):
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
            bloom_level = item.get("bloom_level")
            question_type = item.get("question_type")

            # --- id ---
            if not q_id or not isinstance(q_id, str):
                item_warnings.append(
                    f"{AGENT_NAME}: Item at index {idx} missing 'id' — skipped."
                )
                continue

            q_id = q_id.strip()

            if q_id in seen_ids:
                item_warnings.append(
                    f"{AGENT_NAME}: Duplicate id '{q_id}' — skipped."
                )
                continue
            seen_ids.add(q_id)

            bloom_src = bloom_by_id.get(q_id, {})

            # --- unit: fallback not available from BloomItem (doesn't have unit),
            #     use generated_questions indirectly via bloom (best-effort) ---
            if not unit or not isinstance(unit, str):
                unit = bloom_src.get("unit", f"Unknown Unit")
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'unit' — defaulted to '{unit}'."
                )

            # --- topic ---
            if not topic or not isinstance(topic, str):
                topic = bloom_src.get("topic", "Unknown Topic")
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'topic' — defaulted to '{topic}'."
                )

            # --- question ---
            if not question or not isinstance(question, str):
                question = bloom_src.get("question", "")
                if not question:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' missing 'question' and no fallback — skipped."
                    )
                    continue
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' missing 'question' — restored from bloom_analysis."
                )

            # --- marks ---
            try:
                marks = int(marks)
            except (ValueError, TypeError):
                marks = bloom_src.get("marks")
                if marks is None:
                    item_warnings.append(
                        f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' with no fallback — skipped."
                    )
                    continue
                marks = int(marks)
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' invalid 'marks' — restored from bloom_analysis."
                )
            if marks not in _VALID_MARKS:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' marks={marks} not in {_VALID_MARKS} — skipped."
                )
                continue

            # --- difficulty ---
            if not difficulty or not isinstance(difficulty, str):
                difficulty = bloom_src.get("difficulty", "")
            difficulty = difficulty.strip().lower()
            if difficulty not in _VALID_DIFFICULTIES:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' invalid difficulty '{difficulty}' — skipped."
                )
                continue

            # --- bloom_level: normalise capitalisation ---
            if not bloom_level or not isinstance(bloom_level, str):
                bloom_level = bloom_src.get("bloom_level", "")
            normalised_level = _BLOOM_NORMALISE.get(bloom_level.strip().lower(), "")
            if not normalised_level:
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' unrecognised bloom_level "
                    f"'{bloom_level}' — skipped."
                )
                continue

            # --- question_type: auto-correct from marks ---
            if not question_type or question_type.strip().lower() not in _VALID_QUESTION_TYPES:
                corrected = _MARKS_TO_TYPE[marks]
                item_warnings.append(
                    f"{AGENT_NAME}: Item '{q_id}' invalid question_type "
                    f"'{question_type}' — auto-corrected to '{corrected}'."
                )
                question_type = corrected
            else:
                question_type = question_type.strip().lower()

            validated_questions.append(
                ValidatedQuestion(
                    id=q_id,
                    unit=unit.strip(),
                    topic=topic.strip(),
                    question=question.strip(),
                    marks=marks,
                    difficulty=difficulty,
                    bloom_level=normalised_level,
                    question_type=question_type,
                )
            )

        # Log per-item warnings (non-fatal)
        for warning in item_warnings:
            logger.warning(warning)
        errors.extend(item_warnings)

        # ------------------------------------------------------------------
        # 11. Final guard: at least one validated question must exist
        # ------------------------------------------------------------------
        if not validated_questions:
            error_msg = (
                f"{AGENT_NAME}: No valid questions survived post-LLM validation. "
                "Check the validation prompt or input quality."
            )
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "validated_questions": [],
                "current_agent": AGENT_NAME,
                "status": "failed",
                "errors": errors,
            }

        logger.info(
            f"{AGENT_NAME}: {len(validated_questions)} question(s) validated successfully."
        )

        # ------------------------------------------------------------------
        # 12. Return partial state update
        # ------------------------------------------------------------------
        return {
            "validated_questions": validated_questions,
            "current_agent": AGENT_NAME,
            "status": "running",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------
_validation_agent = ValidationAgent()


def validation_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph-compatible node function for the Validation Agent.

    This is the function registered with StateGraph.add_node().

    Args:
        state: The shared AgentState dict passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    return _validation_agent(state)
