"""
models/state.py

Shared State Object for the Agentic Question Paper Generator.
This TypedDict-based state is passed through every node in the LangGraph workflow.
All agents read from and write to this single shared state.
"""

from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Sub-models as plain typed dicts (LangGraph uses TypedDict, not Pydantic)
# ---------------------------------------------------------------------------

class QuestionItem(TypedDict):
    """Represents a single generated question."""
    id: str
    unit: str
    topic: str
    question: str
    marks: int                  # 2 / 5 / 10 / 15
    difficulty: str             # easy / medium / hard
    question_type: str          # short / long / descriptive


class BloomItem(TypedDict):
    """A question annotated with its Bloom Taxonomy level."""
    id: str
    question: str
    marks: int
    difficulty: str
    bloom_level: str            # Remember / Understand / Apply / Analyze / Evaluate / Create
    bloom_justification: str


class ValidatedQuestion(TypedDict):
    """A question that has passed validation."""
    id: str
    unit: str
    topic: str
    question: str
    marks: int
    difficulty: str
    bloom_level: str
    question_type: str


class AnswerKeyItem(TypedDict):
    """Answer key entry corresponding to a validated question."""
    id: str
    question: str
    marks: int
    model_answer: str
    key_points: List[str]
    marks_breakdown: str        # e.g. "1 + 2 + 2"


class SyllabusTopic(TypedDict):
    """A single unit with its list of topics."""
    unit_number: int
    unit_name: str
    topics: List[str]


class RagChunk(TypedDict):
    """A document chunk produced by the RAG ingestion pipeline."""
    content: str
    source: str
    chunk_id: Any
    first_line: str
    length: int
    file_type: str


class QuestionDistribution(TypedDict):
    """
    Specifies how many questions to generate per marks category
    and the overall difficulty split.
    """
    total_marks: int
    two_mark_questions: int
    five_mark_questions: int
    ten_mark_questions: int
    fifteen_mark_questions: int
    easy_percentage: int        # 0–100
    medium_percentage: int
    hard_percentage: int


class PaperMetadata(TypedDict):
    """Header information printed on the question paper PDF."""
    institution_name: str
    course_name: str
    course_code: str
    semester: str
    exam_type: str              # e.g. "Internal Assessment", "End Semester"
    duration: str               # e.g. "3 Hours"
    maximum_marks: int
    date: Optional[str]


# ---------------------------------------------------------------------------
# Master Workflow State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Central state object shared across all LangGraph nodes.

    Lifecycle:
      rag_chunks  →  syllabus_topics  →  generated_questions
      →  bloom_analysis  →  validated_questions
      →  answer_key  →  final_pdf_path
    """

    # ---- RAG input (from uploaded documents) ----
    rag_chunks: List[RagChunk]                      # All chunks from RAG ingestion
    syllabus_context: Optional[str]                 # Retrieved chunks for syllabus extraction
    content_context: Optional[str]                  # Retrieved chunks for question/answer generation

    # ---- Syllabus Agent output ----
    syllabus_topics: List[SyllabusTopic]            # Structured units & topics

    # ---- Question Generator input params ----
    question_distribution: Optional[QuestionDistribution]

    # ---- Question Generator Agent output ----
    generated_questions: List[QuestionItem]

    # ---- Bloom Taxonomy Agent output ----
    bloom_analysis: List[BloomItem]

    # ---- Validation Agent output ----
    validated_questions: List[ValidatedQuestion]

    # ---- Answer Key Agent output ----
    answer_key: List[AnswerKeyItem]

    # ---- Metadata for PDF header ----
    paper_metadata: Optional[PaperMetadata]

    # ---- PDF Generator output ----
    final_pdf_path: Optional[str]                   # Path to question paper PDF
    answer_key_pdf_path: Optional[str]              # Path to answer key PDF

    # ---- Error tracking ----
    errors: List[str]                               # Accumulated error messages

    # ---- Workflow status ----
    current_agent: Optional[str]                    # Name of the agent currently running
    status: str                                     # initialized / running / completed / failed
