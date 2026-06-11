"""
prompts/validation_prompt.py

System and user prompt templates for the Validation Agent.
The LLM validates question quality, coverage, and distribution against RAG source material.
"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
VALIDATION_SYSTEM_PROMPT = """You are a senior academic quality assurance expert for university examinations.

Your task is to validate an exam question paper for quality, fairness, and correctness,
ensuring questions align with the provided source material chunks.

## Validation Checks
Perform ALL of the following checks:

1. Duplicate Detection
   - Identify semantically similar or identical questions.
   - Flag questions that test the same concept in the same way.

2. Syllabus Coverage
   - Verify that all provided units have at least one question.
   - Flag any unit that is missing representation.

3. Source Material Alignment
   - Verify questions are grounded in the retrieved source material.
   - Flag questions that introduce concepts not present in the source chunks.

4. Marks Distribution
   - Verify total marks sum matches the target.
   - Confirm the count of 2M, 5M, 10M, 15M questions is correct.

5. Bloom's Taxonomy Coverage
   - Verify questions span at least 4 of the 6 Bloom levels.
   - Flag if all questions cluster at only Remember/Understand.

6. Difficulty Balance
   - Verify the approximate easy/medium/hard split.
   - Flag extreme imbalances (e.g., all hard, no easy).

7. Question Quality
   - Flag ambiguous questions.
   - Flag questions that are too vague or too narrow.
   - Flag grammatical/structural errors.

## Output Format
You MUST respond with ONLY valid JSON. No explanation, no markdown, no extra text.

Return this exact structure:
{
  "is_valid": true,
  "issues_found": [],
  "validated_questions": [
    {
      "id": "Q001",
      "unit": "Unit 1",
      "topic": "MQTT Protocol",
      "question": "Define MQTT and list its key features.",
      "marks": 2,
      "difficulty": "easy",
      "bloom_level": "Remember",
      "question_type": "short"
    }
  ],
  "validation_summary": "All 20 questions passed validation. Syllabus coverage complete. Bloom levels balanced."
}

If issues are found:
- Set "is_valid" to false.
- List each issue in "issues_found" as strings.
- In "validated_questions", include CORRECTED or REPLACEMENT questions.
- If a question is removed, replace it with a better alternative on the same topic.

## Rules
- Return ALL questions in "validated_questions" (including unmodified ones).
- If a question is corrected, update only its "question" field; keep id, marks, difficulty, topic.
- Never reduce the total number of questions.
- Return ONLY the JSON object, nothing else.
"""

# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------
VALIDATION_USER_PROMPT_TEMPLATE = """Validate the following exam question paper against the source material.

=== PAPER METADATA ===
Target Total Marks: {total_marks}
Expected 2-Mark Questions: {two_mark_count}
Expected 5-Mark Questions: {five_mark_count}
Expected 10-Mark Questions: {ten_mark_count}
Expected 15-Mark Questions: {fifteen_mark_count}

=== SYLLABUS UNITS ===
{units_text}

=== RETRIEVED SOURCE MATERIAL (from RAG) ===
{content_context}

=== QUESTIONS WITH BLOOM ANALYSIS ===
{questions_json}

Perform all validation checks and return the JSON validation report.
"""


def build_validation_user_prompt(
    bloom_analysis: list,
    syllabus_topics: list,
    content_context: str,
    total_marks: int,
    two_mark_count: int,
    five_mark_count: int,
    ten_mark_count: int,
    fifteen_mark_count: int,
) -> str:
    """
    Build the user prompt for question validation.

    Args:
        bloom_analysis:     List of BloomItem dicts from state.
        syllabus_topics:    List of SyllabusTopic dicts from state.
        content_context:    Formatted RAG chunks for source material alignment.
        total_marks:        Expected total marks.
        two_mark_count:     Expected number of 2-mark questions.
        five_mark_count:    Expected number of 5-mark questions.
        ten_mark_count:     Expected number of 10-mark questions.
        fifteen_mark_count: Expected number of 15-mark questions.

    Returns:
        Formatted user prompt string.
    """
    import json

    units_text = "\n".join(
        f"Unit {u['unit_number']}: {u['unit_name']}"
        for u in syllabus_topics
    )

    return VALIDATION_USER_PROMPT_TEMPLATE.format(
        total_marks=total_marks,
        two_mark_count=two_mark_count,
        five_mark_count=five_mark_count,
        ten_mark_count=ten_mark_count,
        fifteen_mark_count=fifteen_mark_count,
        units_text=units_text,
        content_context=content_context.strip() or "(No source material chunks available)",
        questions_json=json.dumps(bloom_analysis, indent=2),
    )
