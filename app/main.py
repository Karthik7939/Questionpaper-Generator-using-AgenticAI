"""
main.py

FastAPI application entry point for the Agentic Question Paper Generator.

Endpoints:
  POST /generate         — Upload syllabus + parameters → RAG ingest → agents → PDF paths
  GET  /papers           — List all generated PDFs in generated_papers/
  GET  /download/{name}  — Download a generated PDF by filename
  GET  /health           — Health check

The Orchestrator is created once at startup via the lifespan context manager
and shared across all requests.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncGenerator, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.agents.orchestrator import Orchestrator, OrchestratorResult
from app.config import settings
from app.models.state import PaperMetadata, QuestionDistribution
from app.services.logger import setup_logger
from app.services.rag_service import RAGService

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — creates the Orchestrator singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[Orchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup:  instantiate Orchestrator (validates API key, creates dirs)
    Shutdown: log cleanup message
    """
    global _orchestrator
    logger.info("Application starting up...")
    _orchestrator = Orchestrator()
    logger.info("Application ready to serve requests.")
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.api.TITLE,
    description=settings.api.DESCRIPTION,
    version=settings.api.VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""
    success: bool
    message: str
    final_pdf_path: Optional[str] = None
    answer_key_pdf_path: Optional[str] = None
    elapsed_seconds: float
    rag_chunk_count: int = 0
    errors: list[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class RagPreviewResponse(BaseModel):
    """Response body for the /rag/preview endpoint."""
    success: bool
    message: str
    file_name: str
    file_size_mb: float
    chunk_count: int = 0
    debug: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class PaperListResponse(BaseModel):
    """Response body for the /papers endpoint."""
    total: int
    files: list[str]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    version: str
    model: str
    max_upload_size_mb: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check — confirms the service is running."""
    return HealthResponse(
        status="ok",
        version=settings.api.VERSION,
        model=settings.llm.MODEL_NAME,
        max_upload_size_mb=settings.api.MAX_UPLOAD_SIZE_MB,
    )


async def _save_and_validate_upload(file: UploadFile) -> tuple[bytes, str, str, float, Path]:
    """Read, validate, and persist an uploaded file. Returns bytes, name, ext, size_mb, path."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lstrip(".").lower()

    if extension not in settings.api.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '.{extension}'. "
                f"Allowed: {settings.api.ALLOWED_EXTENSIONS}"
            ),
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > settings.api.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {size_mb:.1f} MB exceeds limit of "
                f"{settings.api.MAX_UPLOAD_SIZE_MB} MB. "
                f"Increase MAX_UPLOAD_SIZE_MB in .env if needed."
            ),
        )

    safe_name = Path(filename).name or f"upload_{uuid4().hex}.{extension}"
    upload_path = settings.paths.UPLOADED_DOCUMENTS_DIR / safe_name
    upload_path.write_bytes(file_bytes)
    return file_bytes, filename, extension, size_mb, upload_path


@app.post(
    "/rag/preview",
    response_model=RagPreviewResponse,
    tags=["Debug"],
    summary="Preview RAG chunking for an uploaded file (no generation)",
)
async def rag_preview(
    file: Annotated[UploadFile, File(description="Syllabus PDF or TXT file to preview")],
) -> RagPreviewResponse:
    """
    Run RAG ingestion only and return chunk/debug information.
    Useful for validating uploads before running the full agent pipeline.
    """
    try:
        _file_bytes, filename, _extension, size_mb, upload_path = await _save_and_validate_upload(file)
    except HTTPException:
        raise

    logger.info(f"RAG preview for '{filename}' ({size_mb:.2f} MB)")

    try:
        rag_service = RAGService()
        preview = rag_service.preview_file(str(upload_path))
        return RagPreviewResponse(
            success=True,
            message=f"RAG preview complete — {preview['chunk_count']} chunk(s) created.",
            file_name=Path(filename).name,
            file_size_mb=round(size_mb, 2),
            chunk_count=preview["chunk_count"],
            debug=preview["debug"],
        )
    except Exception as exc:
        logger.error(f"RAG preview failed: {exc}")
        return RagPreviewResponse(
            success=False,
            message=f"RAG preview failed: {exc}",
            file_name=Path(filename).name,
            file_size_mb=round(size_mb, 2),
            errors=[str(exc)],
        )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Generation"],
    summary="Upload syllabus and generate question paper",
)
async def generate_question_paper(
    # --- File upload ---
    file: Annotated[UploadFile, File(description="Syllabus PDF or TXT file")],

    # --- Question distribution (Form fields) ---
    total_marks: Annotated[int, Form(description="Total marks for the paper")] = 100,
    two_mark_questions: Annotated[int, Form(description="Number of 2-mark questions")] = 5,
    five_mark_questions: Annotated[int, Form(description="Number of 5-mark questions")] = 4,
    ten_mark_questions: Annotated[int, Form(description="Number of 10-mark questions")] = 3,
    fifteen_mark_questions: Annotated[int, Form(description="Number of 15-mark questions")] = 2,
    easy_percentage: Annotated[int, Form(description="% of easy questions (0-100)")] = 30,
    medium_percentage: Annotated[int, Form(description="% of medium questions (0-100)")] = 50,
    hard_percentage: Annotated[int, Form(description="% of hard questions (0-100)")] = 20,

    # --- Paper metadata (all optional) ---
    institution_name: Annotated[str, Form()] = "University",
    course_name: Annotated[str, Form()] = "Course",
    course_code: Annotated[str, Form()] = "CS101",
    semester: Annotated[str, Form()] = "I",
    exam_type: Annotated[str, Form()] = "End Semester Examination",
    duration: Annotated[str, Form()] = "3 Hours",
    exam_date: Annotated[Optional[str], Form(description="Optional exam date")] = None,
) -> GenerateResponse:
    """
    Full pipeline endpoint:
    1. Validate and save uploaded file
    2. Ingest via RAG (chunk + embed)
    3. Run Orchestrator (LangGraph workflow + PDF generation)
    4. Return paths to generated PDFs
    """

    # ------------------------------------------------------------------
    # Validate, read, and save upload
    # ------------------------------------------------------------------
    _file_bytes, filename, _extension, size_mb, upload_path = await _save_and_validate_upload(file)

    logger.info(
        f"Received file '{filename}' ({size_mb:.2f} MB), "
        f"saved to '{upload_path}' for RAG ingestion."
    )

    # ------------------------------------------------------------------
    # Build distribution and metadata dicts
    # ------------------------------------------------------------------
    distribution: QuestionDistribution = QuestionDistribution(
        total_marks=total_marks,
        two_mark_questions=two_mark_questions,
        five_mark_questions=five_mark_questions,
        ten_mark_questions=ten_mark_questions,
        fifteen_mark_questions=fifteen_mark_questions,
        easy_percentage=easy_percentage,
        medium_percentage=medium_percentage,
        hard_percentage=hard_percentage,
    )

    paper_metadata: PaperMetadata = PaperMetadata(
        institution_name=institution_name,
        course_name=course_name,
        course_code=course_code,
        semester=semester,
        exam_type=exam_type,
        duration=duration,
        maximum_marks=total_marks,
        date=exam_date,
    )

    # ------------------------------------------------------------------
    # Run Orchestrator (RAG ingest → agents → PDFs)
    # ------------------------------------------------------------------
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator is not initialised. Service is starting up.",
        )

    result: OrchestratorResult = _orchestrator.run(
        uploaded_file_path=str(upload_path),
        distribution=distribution,
        paper_metadata=paper_metadata,
    )

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    if result.success:
        message = (
            f"Question paper and answer key generated successfully "
            f"from {result.rag_chunk_count} RAG chunk(s) in {result.elapsed_seconds:.1f}s."
        )
    else:
        message = (
            f"Generation failed after {result.elapsed_seconds:.1f}s. "
            "See errors for details."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=GenerateResponse(
                success=result.success,
                message=message,
                final_pdf_path=result.final_pdf_path,
                answer_key_pdf_path=result.answer_key_pdf_path,
                elapsed_seconds=result.elapsed_seconds,
                rag_chunk_count=result.rag_chunk_count,
                errors=result.errors,
                debug=result.debug_info,
            ).model_dump(),
        )

    return GenerateResponse(
        success=result.success,
        message=message,
        final_pdf_path=result.final_pdf_path,
        answer_key_pdf_path=result.answer_key_pdf_path,
        elapsed_seconds=result.elapsed_seconds,
        rag_chunk_count=result.rag_chunk_count,
        errors=result.errors,
        debug=result.debug_info,
    )


@app.get(
    "/papers",
    response_model=PaperListResponse,
    tags=["Files"],
    summary="List all generated papers and answer keys",
)
async def list_papers() -> PaperListResponse:
    """Return a list of all PDF files in the generated_papers/ directory."""
    output_dir: Path = settings.paths.GENERATED_PAPERS_DIR
    pdf_files = sorted(
        [f.name for f in output_dir.glob("*.pdf")],
        reverse=True,
    )
    return PaperListResponse(total=len(pdf_files), files=pdf_files)


@app.get(
    "/download/{filename}",
    tags=["Files"],
    summary="Download a generated PDF by filename",
)
async def download_paper(filename: str) -> FileResponse:
    """
    Stream a generated PDF file for download.

    Args:
        filename: Name of the PDF file (e.g., question_paper_20240611_143022.pdf)
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    filepath = settings.paths.GENERATED_PAPERS_DIR / filename

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found.",
        )

    return FileResponse(
        path=str(filepath),
        media_type="application/pdf",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        reload=settings.api.DEBUG,
        log_level=settings.log.LOG_LEVEL.lower(),
    )
