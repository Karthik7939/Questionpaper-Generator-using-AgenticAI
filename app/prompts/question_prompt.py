"""
prompts/question_prompt.py

System and user prompt templates for the Question Generator Agent.
The LLM generates university-style questions from syllabus topics and RAG content chunks.
"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
QUESTION_SYSTEM_PROMPT = """You are an expert university examination paper setter with 20+ years of experience.

Your task is to generate high-quality university-level exam questions from provided syllabus topics
and retrieved source material chunks.

## Question Types & Marks
- 2 Marks  → Short answer (definition, one-line explanation, fill in blank)
- 5 Marks  → Brief essay (explain with examples, compare concepts)
- 10 Marks → Long answer (detailed explanation with diagrams/steps)
- 15 Marks → Essay/Case study (comprehensive analysis, design, application)

## Difficulty Levels
- easy:   Recall and basic understanding. Suitable for all students.
- medium: Application and analysis. Requires deeper understanding.
- hard:   Evaluation and synthesis. Requires critical thinking.

## Output Format
You MUST respond with ONLY a valid JSON array. No explanation, no markdown, no extra text.

Return this exact structure:
[
  {
    "id": "Q001",
    "unit": "Unit 1",
    "topic": "MQTT Protocol",
    "question": "Define MQTT and list its key features.",
    "marks": 2,
    "difficulty": "easy",
    "question_type": "short"
  },
  {
    "id": "Q002",
    "unit": "Unit 2",
    "topic": "IoT Architecture",
    "question": "Explain the three-layer IoT architecture with a neat diagram.",
    "marks": 10,
    "difficulty": "medium",
    "question_type": "long"
  }
]

## Rules
- Generate ONLY the number of questions requested per marks category.
- Base questions on the retrieved source material — stay faithful to the course content.
- Spread questions across ALL units and topics proportionally.
- NO duplicate questions (check for semantic similarity, not just exact match).
- Questions must be university-level (not trivial or overly vague).
- question_type values: "short" (2M), "brief" (5M), "long" (10M), "essay" (15M).
- IDs must be sequential: Q001, Q002, Q003, ...
- difficulty must be exactly: "easy", "medium", or "hard" (lowercase).
- Return ONLY the JSON array, nothing else.
"""

# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------
QUESTION_USER_PROMPT_TEMPLATE = """Generate exam questions based on the following syllabus, source material, and requirements.

=== SYLLABUS TOPICS ===
{syllabus_topics_text}

=== RETRIEVED SOURCE MATERIAL (from RAG) ===
{content_context}

=== QUESTION REQUIREMENTS ===
Total Marks: {total_marks}
- 2 Mark Questions: {two_mark_count} questions
- 5 Mark Questions: {five_mark_count} questions
- 10 Mark Questions: {ten_mark_count} questions
- 15 Mark Questions: {fifteen_mark_count} questions

Difficulty Distribution:
- Easy: {easy_pct}%
- Medium: {medium_pct}%
- Hard: {hard_pct}%

Generate exactly the number of questions specified above.
Ensure all units are covered proportionally.
Base each question on concepts found in the retrieved source material.
Return a JSON array of question objects.
"""


def build_question_user_prompt(
    syllabus_topics: list,
    content_context: str,
    total_marks: int,
    two_mark_count: int,
    five_mark_count: int,
    ten_mark_count: int,
    fifteen_mark_count: int,
    easy_pct: int,
    medium_pct: int,
    hard_pct: int,
) -> str:
    """
    Build the user prompt for question generation.

    Args:
        syllabus_topics:    List of SyllabusTopic dicts from state.
        content_context:    Formatted RAG chunks for source material.
        total_marks:        Total marks for the paper.
        two_mark_count:     Number of 2-mark questions.
        five_mark_count:    Number of 5-mark questions.
        ten_mark_count:     Number of 10-mark questions.
        fifteen_mark_count: Number of 15-mark questions.
        easy_pct:           Percentage of easy questions (0–100).
        medium_pct:         Percentage of medium questions (0–100).
        hard_pct:           Percentage of hard questions (0–100).

    Returns:
        Formatted user prompt string.
    """
    topics_lines = []
    for unit in syllabus_topics:
        topics_lines.append(
            f"Unit {unit['unit_number']}: {unit['unit_name']}"
        )
        for topic in unit["topics"]:
            topics_lines.append(f"  - {topic}")

    return QUESTION_USER_PROMPT_TEMPLATE.format(
        syllabus_topics_text="\n".join(topics_lines),
        content_context=content_context.strip() or "(No source material chunks available)",
        total_marks=total_marks,
        two_mark_count=two_mark_count,
        five_mark_count=five_mark_count,
        ten_mark_count=ten_mark_count,
        fifteen_mark_count=fifteen_mark_count,
        easy_pct=easy_pct,
        medium_pct=medium_pct,
        hard_pct=hard_pct,
    )
