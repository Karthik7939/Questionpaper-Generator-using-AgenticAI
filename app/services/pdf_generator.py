"""
services/pdf_generator.py

PDF Generator Service for the Agentic Question Paper Generator.

Generates two professional A4 PDFs using ReportLab Platypus (flowable-based):
  1. Question Paper PDF — questions grouped by section (2M / 5M / 10M / 15M)
  2. Answer Key PDF    — model answers, key points, and marks breakdown

Stored in: generated_papers/
"""

import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings
from app.models.state import AnswerKeyItem, PaperMetadata, ValidatedQuestion
from app.services.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    """Build and return the paragraph style dictionary."""
    base = getSampleStyleSheet()

    styles = {
        "institution": ParagraphStyle(
            "institution",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "course_name": ParagraphStyle(
            "course_name",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "header_info": ParagraphStyle(
            "header_info",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section_title": ParagraphStyle(
            "section_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1a1a2e"),
        ),
        "question": ParagraphStyle(
            "question",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=8,
            leading=14,
        ),
        "answer_heading": ParagraphStyle(
            "answer_heading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "answer_body": ParagraphStyle(
            "answer_body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=4,
            leading=14,
        ),
        "key_point": ParagraphStyle(
            "key_point",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            alignment=TA_LEFT,
            leftIndent=14,
            spaceAfter=2,
            leading=13,
        ),
        "marks_breakdown": ParagraphStyle(
            "marks_breakdown",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            alignment=TA_LEFT,
            leftIndent=14,
            spaceAfter=6,
            textColor=colors.HexColor("#555555"),
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#888888"),
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# PDFGenerator class
# ---------------------------------------------------------------------------

class PDFGenerator:
    """
    Generates question paper and answer key PDFs.

    Usage:
        generator = PDFGenerator()
        paper_path = generator.generate_question_paper(questions, metadata)
        key_path   = generator.generate_answer_key(answer_key, metadata)
    """

    def __init__(self) -> None:
        settings.paths.ensure_directories()
        self.output_dir: Path = settings.paths.GENERATED_PAPERS_DIR
        self.styles = _build_styles()
        logger.info("PDFGenerator initialized.")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_question_paper(
        self,
        validated_questions: list[ValidatedQuestion],
        paper_metadata: Optional[PaperMetadata] = None,
    ) -> str:
        """
        Generate a professional question paper PDF.

        Questions are grouped into sections by marks category:
          Section A — 2 Marks
          Section B — 5 Marks
          Section C — 10 Marks
          Section D — 15 Marks

        Args:
            validated_questions: List of ValidatedQuestion dicts from state.
            paper_metadata:      Optional header metadata for the PDF.

        Returns:
            Absolute path string of the generated PDF file.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"question_paper_{timestamp}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            leftMargin=settings.pdf.MARGIN_LEFT,
            rightMargin=settings.pdf.MARGIN_RIGHT,
            topMargin=settings.pdf.MARGIN_TOP,
            bottomMargin=settings.pdf.MARGIN_BOTTOM,
        )

        story = []
        story.extend(self._build_header(paper_metadata))
        story.extend(self._build_instructions())
        story.extend(self._build_question_sections(validated_questions))
        story.extend(self._build_footer())

        doc.build(story)
        logger.info(f"Question paper PDF generated: {filepath}")
        return str(filepath)

    def generate_answer_key(
        self,
        answer_key: list[AnswerKeyItem],
        paper_metadata: Optional[PaperMetadata] = None,
    ) -> str:
        """
        Generate a professional answer key PDF for examiners.

        Args:
            answer_key:     List of AnswerKeyItem dicts from state.
            paper_metadata: Optional header metadata for the PDF.

        Returns:
            Absolute path string of the generated PDF file.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answer_key_{timestamp}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            leftMargin=settings.pdf.MARGIN_LEFT,
            rightMargin=settings.pdf.MARGIN_RIGHT,
            topMargin=settings.pdf.MARGIN_TOP,
            bottomMargin=settings.pdf.MARGIN_BOTTOM,
        )

        story = []
        story.extend(self._build_header(paper_metadata, is_answer_key=True))
        story.extend(self._build_answer_key_body(answer_key))
        story.extend(self._build_footer())

        doc.build(story)
        logger.info(f"Answer key PDF generated: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Header builder
    # ------------------------------------------------------------------

    def _build_header(
        self,
        metadata: Optional[PaperMetadata],
        is_answer_key: bool = False,
    ) -> list:
        """Build the institution/exam header flowables."""
        story = []

        if metadata:
            story.append(
                Paragraph(metadata["institution_name"], self.styles["institution"])
            )
            story.append(
                Paragraph(
                    f"{metadata['course_name']} ({metadata['course_code']})",
                    self.styles["course_name"],
                )
            )

            label = "ANSWER KEY" if is_answer_key else metadata["exam_type"].upper()
            story.append(
                Paragraph(label, self.styles["course_name"])
            )

            # Two-column info row: Semester | Duration / Max Marks
            info_left = (
                f"Semester: {metadata['semester']}"
                + (f" &nbsp;&nbsp; Date: {metadata['date']}" if metadata.get("date") else "")
            )
            info_right = (
                f"Duration: {metadata['duration']} &nbsp;&nbsp; "
                f"Max. Marks: {metadata['maximum_marks']}"
            )

            info_table = Table(
                [[Paragraph(info_left, self.styles["header_info"]),
                  Paragraph(info_right, self.styles["header_info"])]],
                colWidths=[
                    (settings.pdf.PAGE_WIDTH
                     - settings.pdf.MARGIN_LEFT
                     - settings.pdf.MARGIN_RIGHT) / 2,
                    (settings.pdf.PAGE_WIDTH
                     - settings.pdf.MARGIN_LEFT
                     - settings.pdf.MARGIN_RIGHT) / 2,
                ],
            )
            info_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(Spacer(1, 6))
            story.append(info_table)
        else:
            title = "ANSWER KEY" if is_answer_key else "EXAMINATION QUESTION PAPER"
            story.append(Paragraph(title, self.styles["institution"]))

        story.append(Spacer(1, 4))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=colors.HexColor("#1a1a2e"),
            )
        )
        story.append(Spacer(1, 8))
        return story

    # ------------------------------------------------------------------
    # Exam instructions
    # ------------------------------------------------------------------

    def _build_instructions(self) -> list:
        """Build standard exam instructions block."""
        instructions = [
            "Answer ALL questions.",
            "Each question carries marks as indicated.",
            "Write answers in the answer booklet provided.",
            "Figures in parentheses indicate maximum marks.",
        ]
        story = [Paragraph("<b>Instructions:</b>", self.styles["answer_heading"])]
        for instr in instructions:
            story.append(Paragraph(f"• {instr}", self.styles["key_point"]))
        story.append(Spacer(1, 10))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"))
        )
        return story

    # ------------------------------------------------------------------
    # Question sections builder
    # ------------------------------------------------------------------

    def _build_question_sections(
        self, validated_questions: list[ValidatedQuestion]
    ) -> list:
        """
        Group questions into sections A/B/C/D by marks category
        and build the flowable list.
        """
        sections = [
            (2,  "Section A — Short Answer Questions (2 Marks each)"),
            (5,  "Section B — Brief Answer Questions (5 Marks each)"),
            (10, "Section C — Long Answer Questions (10 Marks each)"),
            (15, "Section D — Essay / Case Study Questions (15 Marks each)"),
        ]

        # Group questions by marks
        by_marks: dict[int, list[ValidatedQuestion]] = {2: [], 5: [], 10: [], 15: []}
        for q in validated_questions:
            marks = q.get("marks")
            if marks in by_marks:
                by_marks[marks].append(q)

        story = []
        q_number = 1  # Global question counter across all sections

        for marks_value, section_label in sections:
            qs = by_marks.get(marks_value, [])
            if not qs:
                continue

            story.append(Paragraph(section_label, self.styles["section_title"]))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#999999"),
                )
            )
            story.append(Spacer(1, 6))

            for q in qs:
                q_text = (
                    f"<b>Q{q_number}.</b> {q['question']}"
                    f" &nbsp;&nbsp; <b>[{q['marks']} Marks]</b>"
                )
                story.append(Paragraph(q_text, self.styles["question"]))
                q_number += 1

            story.append(Spacer(1, 8))

        return story

    # ------------------------------------------------------------------
    # Answer key body builder
    # ------------------------------------------------------------------

    def _build_answer_key_body(self, answer_key: list[AnswerKeyItem]) -> list:
        """Build the flowable list for the answer key document."""
        story = []
        story.append(
            Paragraph(
                "<b>MODEL ANSWERS AND MARKING SCHEME</b>",
                self.styles["section_title"],
            )
        )
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"))
        )
        story.append(Spacer(1, 8))

        for idx, item in enumerate(answer_key, start=1):
            # --- Question heading ---
            q_heading = (
                f"<b>Q{idx}. ({item['marks']} Marks)</b>  {item['question']}"
            )
            story.append(Paragraph(q_heading, self.styles["answer_heading"]))

            # --- Model answer ---
            story.append(
                Paragraph(
                    f"<b>Model Answer:</b> {item['model_answer']}",
                    self.styles["answer_body"],
                )
            )

            # --- Key points ---
            story.append(Paragraph("<b>Key Points:</b>", self.styles["answer_body"]))
            for kp in item["key_points"]:
                story.append(Paragraph(f"• {kp}", self.styles["key_point"]))

            # --- Marks breakdown ---
            story.append(
                Paragraph(
                    f"<i>Marks Breakdown: {item['marks_breakdown']}</i>",
                    self.styles["marks_breakdown"],
                )
            )

            # Separator between answers
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.3,
                    color=colors.HexColor("#dddddd"),
                )
            )

        return story

    # ------------------------------------------------------------------
    # Footer builder
    # ------------------------------------------------------------------

    def _build_footer(self) -> list:
        """Build the document footer flowables."""
        return [
            Spacer(1, 16),
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#cccccc"),
            ),
            Spacer(1, 4),
            Paragraph(settings.pdf.FOOTER_TEXT, self.styles["footer"]),
        ]
