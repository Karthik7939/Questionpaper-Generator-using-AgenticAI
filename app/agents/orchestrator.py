"""
agents/orchestrator.py

Orchestrator for the Agentic Question Paper Generator.

The Orchestrator is the single entry point for all external callers
(FastAPI endpoints, CLI tools, tests). It:

  1. Accepts an uploaded file path and parameters.
  2. Ingests the document through the RAG pipeline (chunk + embed).
  3. Retrieves context chunks for downstream agents.
  4. Builds the initial AgentState.
  5. Invokes the compiled LangGraph workflow.
  6. Triggers PDF generation (question paper + answer key).
  7. Returns a structured OrchestratorResult.

No business logic lives here — it only coordinates the other components.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models.state import AgentState, PaperMetadata, QuestionDistribution
from app.services.logger import setup_logger
from app.services.pdf_generator import PDFGenerator
from app.services.rag_service import RAGService
from app.workflows.langgraph_workflow import build_initial_state, create_workflow

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass returned to callers
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """
    Structured result returned by the Orchestrator after a workflow run.

    Attributes:
        success:              True if the workflow and PDF generation completed.
        final_pdf_path:       Absolute path to the question paper PDF (or None).
        answer_key_pdf_path:  Absolute path to the answer key PDF (or None).
        errors:               List of all errors and warnings accumulated during the run.
        elapsed_seconds:      Total wall-clock time for the full pipeline in seconds.
        final_state:          The raw final AgentState dict (for debugging/testing).
        rag_chunk_count:      Number of chunks produced by RAG ingestion.
        debug_info:           RAG and pipeline debug metadata for the frontend.
    """
    success: bool
    final_pdf_path: Optional[str]
    answer_key_pdf_path: Optional[str]
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    final_state: Optional[dict[str, Any]] = None
    rag_chunk_count: int = 0
    debug_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Top-level coordinator for the Question Paper Generation pipeline.

    Usage:
        orchestrator = Orchestrator()
        result = orchestrator.run(
            uploaded_file_paths=["/path/to/syllabus.pdf", "/path/to/module2.pdf"],
            distribution={...},
            paper_metadata={...},
        )
    """

    def __init__(self) -> None:
        logger.info("Orchestrator initializing...")
        settings.paths.ensure_directories()
        self._workflow = create_workflow()
        self._pdf_generator = PDFGenerator()
        logger.info("Orchestrator ready.")

    def run(
        self,
        uploaded_file_paths: list[str],
        distribution: QuestionDistribution,
        paper_metadata: Optional[PaperMetadata] = None,
    ) -> OrchestratorResult:
        """
        Execute the full question paper generation pipeline.

        Steps:
          1. Validate inputs
          2. Ingest documents via RAG (chunk + embed)
          3. Retrieve context chunks for agents
          4. Build initial AgentState
          5. Invoke LangGraph workflow
          6. Generate PDFs if workflow succeeded
          7. Return OrchestratorResult

        Args:
            uploaded_file_paths: Paths to uploaded syllabus/course documents.
            distribution:        Question distribution parameters.
            paper_metadata:      Optional PDF header metadata.

        Returns:
            OrchestratorResult with paths, errors, and timing.
        """
        start_time = time.perf_counter()
        logger.info(
            f"Orchestrator: Starting question paper generation pipeline "
            f"with {len(uploaded_file_paths)} file(s)."
        )

        # ------------------------------------------------------------------
        # 1. Validate inputs before entering the workflow
        # ------------------------------------------------------------------
        validation_errors = self._validate_inputs(uploaded_file_paths, distribution)
        if validation_errors:
            elapsed = time.perf_counter() - start_time
            for err in validation_errors:
                logger.error(f"Orchestrator input error: {err}")
            return OrchestratorResult(
                success=False,
                final_pdf_path=None,
                answer_key_pdf_path=None,
                errors=validation_errors,
                elapsed_seconds=elapsed,
            )

        # ------------------------------------------------------------------
        # 2. Ingest via RAG and prepare agent contexts
        # ------------------------------------------------------------------
        rag_chunk_count = 0
        rag_debug: dict[str, Any] = {}
        try:
            rag_service = RAGService()
            rag_chunk_count = rag_service.ingest_files(uploaded_file_paths)
            if rag_chunk_count == 0:
                raise RuntimeError("RAG ingestion produced no chunks from the uploaded file(s).")
            contexts = rag_service.prepare_agent_contexts(file_count=len(uploaded_file_paths))
            rag_debug = contexts.get("debug", {})
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = f"Orchestrator: RAG ingestion failed — {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            return OrchestratorResult(
                success=False,
                final_pdf_path=None,
                answer_key_pdf_path=None,
                errors=[error_msg],
                elapsed_seconds=elapsed,
                rag_chunk_count=rag_chunk_count,
                debug_info=self._build_debug_info(rag_debug, {}, uploaded_file_paths),
            )

        # ------------------------------------------------------------------
        # 3. Build initial state
        # ------------------------------------------------------------------
        initial_state: AgentState = build_initial_state(
            rag_chunks=contexts["rag_chunks"],
            syllabus_context=contexts["syllabus_context"],
            content_context=contexts["content_context"],
            distribution=distribution,
            paper_metadata=paper_metadata,
        )
        logger.info(
            f"Orchestrator: Initial state built from {rag_chunk_count} RAG chunk(s). "
            f"Distribution — total_marks={distribution['total_marks']}, "
            f"2M×{distribution['two_mark_questions']}, "
            f"5M×{distribution['five_mark_questions']}, "
            f"10M×{distribution['ten_mark_questions']}, "
            f"15M×{distribution['fifteen_mark_questions']}"
        )

        # ------------------------------------------------------------------
        # 4. Invoke LangGraph workflow
        # ------------------------------------------------------------------
        try:
            logger.info("Orchestrator: Invoking LangGraph workflow...")
            final_state: dict[str, Any] = self._workflow.invoke(initial_state)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = f"Orchestrator: Workflow invocation failed — {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            return OrchestratorResult(
                success=False,
                final_pdf_path=None,
                answer_key_pdf_path=None,
                errors=[error_msg],
                elapsed_seconds=elapsed,
                rag_chunk_count=rag_chunk_count,
                debug_info=self._build_debug_info(rag_debug, {}, uploaded_file_paths),
            )

        # ------------------------------------------------------------------
        # 5. Check workflow outcome
        # ------------------------------------------------------------------
        status: str = final_state.get("status", "failed")
        accumulated_errors: list[str] = final_state.get("errors", [])

        if status == "failed":
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Orchestrator: Workflow completed with status 'failed'. "
                f"Errors: {len(accumulated_errors)}"
            )
            return OrchestratorResult(
                success=False,
                final_pdf_path=None,
                answer_key_pdf_path=None,
                errors=accumulated_errors,
                elapsed_seconds=elapsed,
                final_state=final_state,
                rag_chunk_count=rag_chunk_count,
                debug_info=self._build_debug_info(rag_debug, final_state, uploaded_file_paths),
            )

        logger.info("Orchestrator: Workflow completed successfully. Generating PDFs...")

        # ------------------------------------------------------------------
        # 6. Generate question paper PDF
        # ------------------------------------------------------------------
        final_pdf_path: Optional[str] = None
        answer_key_pdf_path: Optional[str] = None
        pdf_errors: list[str] = []

        validated_questions = final_state.get("validated_questions", [])
        answer_key = final_state.get("answer_key", [])

        try:
            if validated_questions:
                final_pdf_path = self._pdf_generator.generate_question_paper(
                    validated_questions=validated_questions,
                    paper_metadata=paper_metadata,
                )
                logger.info(f"Orchestrator: Question paper PDF → {final_pdf_path}")
            else:
                pdf_errors.append(
                    "Orchestrator: No validated_questions found; question paper PDF not generated."
                )
        except Exception as exc:
            error_msg = f"Orchestrator: Question paper PDF generation failed — {exc}"
            logger.error(error_msg)
            pdf_errors.append(error_msg)

        # ------------------------------------------------------------------
        # 7. Generate answer key PDF
        # ------------------------------------------------------------------
        try:
            if answer_key:
                answer_key_pdf_path = self._pdf_generator.generate_answer_key(
                    answer_key=answer_key,
                    paper_metadata=paper_metadata,
                )
                logger.info(f"Orchestrator: Answer key PDF → {answer_key_pdf_path}")
            else:
                pdf_errors.append(
                    "Orchestrator: No answer_key found; answer key PDF not generated."
                )
        except Exception as exc:
            error_msg = f"Orchestrator: Answer key PDF generation failed — {exc}"
            logger.error(error_msg)
            pdf_errors.append(error_msg)

        # ------------------------------------------------------------------
        # 8. Determine overall success and return result
        # ------------------------------------------------------------------
        all_errors = accumulated_errors + pdf_errors
        success = final_pdf_path is not None

        elapsed = time.perf_counter() - start_time

        if success:
            logger.info(
                f"Orchestrator: Pipeline completed successfully in {elapsed:.2f}s. "
                f"Errors/warnings: {len(all_errors)}"
            )
        else:
            logger.error(
                f"Orchestrator: Pipeline finished but PDF generation failed "
                f"after {elapsed:.2f}s."
            )

        return OrchestratorResult(
            success=success,
            final_pdf_path=final_pdf_path,
            answer_key_pdf_path=answer_key_pdf_path,
            errors=all_errors,
            elapsed_seconds=elapsed,
            final_state=final_state,
            rag_chunk_count=rag_chunk_count,
            debug_info=self._build_debug_info(rag_debug, final_state, uploaded_file_paths),
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_debug_info(
        rag_debug: dict[str, Any],
        final_state: dict[str, Any],
        uploaded_file_paths: list[str],
    ) -> dict[str, Any]:
        """Summarise RAG and agent pipeline state for the frontend debug panel."""
        syllabus_topics = final_state.get("syllabus_topics", [])
        generated = final_state.get("generated_questions", [])
        validated = final_state.get("validated_questions", [])
        answer_key = final_state.get("answer_key", [])

        return {
            "uploaded_files": [Path(p).name for p in uploaded_file_paths],
            "file_count": len(uploaded_file_paths),
            "rag": rag_debug,
            "pipeline": {
                "status": final_state.get("status"),
                "current_agent": final_state.get("current_agent"),
                "syllabus_units": len(syllabus_topics),
                "syllabus_topics_preview": [
                    {
                        "unit_number": u.get("unit_number"),
                        "unit_name": u.get("unit_name"),
                        "topic_count": len(u.get("topics", [])),
                        "topics": u.get("topics", [])[:5],
                    }
                    for u in syllabus_topics[:10]
                ],
                "generated_questions": len(generated),
                "validated_questions": len(validated),
                "answer_key_entries": len(answer_key),
                "error_count": len(final_state.get("errors", [])),
            },
        }

    # ------------------------------------------------------------------
    # Input validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        uploaded_file_paths: list[str],
        distribution: QuestionDistribution,
    ) -> list[str]:
        """
        Validate the inputs before starting the workflow.

        Returns:
            A list of error strings. Empty list means inputs are valid.
        """
        errors: list[str] = []

        if not uploaded_file_paths:
            errors.append("No files uploaded. Please upload at least one syllabus document.")
        else:
            for file_path in uploaded_file_paths:
                if not file_path or not str(file_path).strip():
                    errors.append("One of the uploaded file paths is empty.")
                elif not Path(file_path).is_file():
                    errors.append(f"Uploaded file not found: '{file_path}'.")

        if not isinstance(distribution, dict):
            errors.append("question_distribution must be a dict.")
            return errors

        required_int_fields = [
            "total_marks",
            "two_mark_questions",
            "five_mark_questions",
            "ten_mark_questions",
            "fifteen_mark_questions",
            "easy_percentage",
            "medium_percentage",
            "hard_percentage",
        ]
        for field_name in required_int_fields:
            value = distribution.get(field_name)
            if value is None:
                errors.append(f"question_distribution missing required field: '{field_name}'.")
            elif not isinstance(value, int) or value < 0:
                errors.append(
                    f"question_distribution['{field_name}'] must be a non-negative integer, "
                    f"got: {value!r}"
                )

        easy = distribution.get("easy_percentage", 0)
        medium = distribution.get("medium_percentage", 0)
        hard = distribution.get("hard_percentage", 0)
        if isinstance(easy, int) and isinstance(medium, int) and isinstance(hard, int):
            total_pct = easy + medium + hard
            if total_pct != 100:
                errors.append(
                    f"easy_percentage + medium_percentage + hard_percentage must equal 100, "
                    f"got: {total_pct}"
                )

        two_m = distribution.get("two_mark_questions", 0)
        five_m = distribution.get("five_mark_questions", 0)
        ten_m = distribution.get("ten_mark_questions", 0)
        fifteen_m = distribution.get("fifteen_mark_questions", 0)
        total_marks = distribution.get("total_marks", 0)

        if all(isinstance(v, int) for v in [two_m, five_m, ten_m, fifteen_m, total_marks]):
            computed = (two_m * 2) + (five_m * 5) + (ten_m * 10) + (fifteen_m * 15)
            if computed != total_marks:
                errors.append(
                    f"total_marks ({total_marks}) does not match the sum of question marks "
                    f"({two_m}×2 + {five_m}×5 + {ten_m}×10 + {fifteen_m}×15 = {computed})."
                )

        return errors
