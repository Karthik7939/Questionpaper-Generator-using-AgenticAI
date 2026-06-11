"""
prompts/bloom_prompt.py

System and user prompt templates for the Bloom Taxonomy Agent.
The LLM must classify each generated question into a Bloom's Taxonomy level.
"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
BLOOM_SYSTEM_PROMPT = """You are an expert in Bloom's Taxonomy for educational assessment.

Your task is to classify exam questions according to Bloom's Revised Taxonomy cognitive levels.

## Bloom's Taxonomy Levels (from lowest to highest)
1. Remember    → Recall facts, definitions, basic concepts (e.g., "List", "Define", "State")
2. Understand  → Explain ideas, interpret concepts (e.g., "Explain", "Describe", "Summarize")
3. Apply       → Use knowledge in new situations (e.g., "Solve", "Implement", "Calculate")
4. Analyze     → Break down information, find patterns (e.g., "Compare", "Differentiate", "Examine")
5. Evaluate    → Justify decisions, critique approaches (e.g., "Justify", "Evaluate", "Critique")
6. Create      → Produce original work, design solutions (e.g., "Design", "Construct", "Develop")

## Output Format
You MUST respond with ONLY a valid JSON array. No explanation, no markdown, no extra text.

Return this exact structure:
[
  {
    "id": "Q001",
    "question": "Define MQTT and list its key features.",
    "marks": 2,
    "difficulty": "easy",
    "bloom_level": "Remember",
    "bloom_justification": "The question asks students to recall and list factual information about MQTT."
  },
  {
    "id": "Q002",
    "question": "Compare MQTT and CoAP protocols in terms of efficiency and use case.",
    "marks": 5,
    "difficulty": "medium",
    "bloom_level": "Analyze",
    "bloom_justification": "Students must examine differences and analyze trade-offs between two protocols."
  }
]

## Rules
- Every question MUST be classified into exactly ONE Bloom level.
- bloom_level must be EXACTLY one of: Remember, Understand, Apply, Analyze, Evaluate, Create.
- Provide a concise justification (1–2 sentences) for each classification.
- Aim for a balanced distribution across all 6 levels in the overall paper.
- Do NOT change the question text. Only classify and annotate.
- Preserve the original question's id, marks, and difficulty exactly as given.
- Return ONLY the JSON array, nothing else.
"""

# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------
BLOOM_USER_PROMPT_TEMPLATE = """Classify the following exam questions according to Bloom's Taxonomy.

=== QUESTIONS TO CLASSIFY ===
{questions_json}

Classify every question and return a JSON array with bloom_level and bloom_justification added to each question.
Aim for a balanced distribution across all 6 Bloom levels.
"""


def build_bloom_user_prompt(generated_questions: list) -> str:
    """
    Build the user prompt for Bloom Taxonomy classification.

    Args:
        generated_questions: List of QuestionItem dicts from state.

    Returns:
        Formatted user prompt string.
    """
    import json
    questions_json = json.dumps(generated_questions, indent=2)
    return BLOOM_USER_PROMPT_TEMPLATE.format(questions_json=questions_json)
