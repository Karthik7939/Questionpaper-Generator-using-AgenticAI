# Folder Structure

```
question-paper-generator/

├── app/
├── uploaded_documents/
├── generated_papers/
├── logs/
├── tests/
├── requirements.txt
└── README.md
```

---

## app/

Contains the complete backend implementation.

### agents/

Implements all AI agents responsible for syllabus analysis, question generation, validation, Bloom's taxonomy classification, and answer key generation.

### workflows/

Contains the LangGraph workflow definition responsible for orchestrating the complete pipeline.

### prompts/

Stores prompt templates used by each AI agent.

### services/

Contains reusable services such as:

- Groq LLM wrapper
- PDF generation
- Logging

### models/

Defines the LangGraph state shared between all agents.

---

## uploaded_documents/

Stores uploaded syllabus files temporarily.

---

## generated_papers/

Stores generated PDFs.

---

## logs/

Contains application log files.

---

## tests/

Contains unit and integration tests.