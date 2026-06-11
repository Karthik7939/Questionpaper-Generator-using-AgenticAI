# Agentic AI Question Paper Generator

An AI-powered question paper generator that automatically creates university/college-level question papers from uploaded syllabus documents using a **Multi-Agent AI architecture** built with LangGraph and Groq LLM.

---

## Features

- Upload a syllabus PDF or TXT and get a full question paper in seconds
- Multi-Agent pipeline: Syllabus → Questions → Bloom's Taxonomy → Validation → Answer Key
- Professionally formatted PDF output (question paper + answer key)
- Configurable marks distribution (2M / 5M / 10M / 15M questions)
- Difficulty balancing (Easy / Medium / Hard percentages)
- Bloom's Taxonomy classification for every question
- Automatic validation and correction of question quality
- FastAPI REST API with file upload and PDF download
- Rotating file logger with color console output

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+ |
| AI Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM Provider | Groq API |
| Default Model | `llama-3.3-70b-versatile` |
| PDF Generation | ReportLab |
| PDF Parsing | pypdf |
| API Framework | FastAPI + Uvicorn |
| Testing | pytest |

---

## Project Structure

```
question-paper-generator/
│
├── app/
│   ├── main.py                        # FastAPI application entry point
│   ├── config.py                      # Centralised settings (LLM, paths, PDF, API)
│   │
│   ├── agents/
│   │   ├── orchestrator.py            # Top-level coordinator
│   │   ├── syllabus_agent.py          # Extracts units & topics from syllabus
│   │   ├── question_generator_agent.py# Generates exam questions
│   │   ├── bloom_agent.py             # Classifies questions by Bloom's Taxonomy
│   │   ├── validation_agent.py        # Validates and corrects question quality
│   │   └── answerkey_agent.py         # Generates model answers & marking schemes
│   │
│   ├── workflows/
│   │   └── langgraph_workflow.py      # LangGraph StateGraph definition
│   │
│   ├── prompts/
│   │   ├── syllabus_prompt.py         # Syllabus extraction prompt
│   │   ├── question_prompt.py         # Question generation prompt
│   │   ├── bloom_prompt.py            # Bloom classification prompt
│   │   ├── validation_prompt.py       # Validation prompt
│   │   └── answerkey_prompt.py        # Answer key prompt
│   │
│   ├── services/
│   │   ├── llm_service.py             # Groq LLM wrapper with retries
│   │   ├── pdf_generator.py           # ReportLab PDF generation service
│   │   └── logger.py                  # Rotating file + color console logger
│   │
│   └── models/
│       └── state.py                   # LangGraph TypedDict state definition
│
├── uploaded_documents/                # Store uploaded syllabus files
├── generated_papers/                  # Output: question paper & answer key PDFs
├── logs/                              # application.log (auto-created)
├── tests/                             # Test suites
├── .env                               # Environment variables (see below)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd question-paper-generator
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the `.env` file and set your Groq API key:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (defaults shown)
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.3
MAX_RETRIES=3
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

Get your free Groq API key at: https://console.groq.com

---

## Running the Server

```bash
python -m app.main
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /generate` — Generate a Question Paper

Upload a syllabus and generate a full question paper.

**Form Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | — | Syllabus PDF or TXT (required) |
| `total_marks` | int | 100 | Total marks for the paper |
| `two_mark_questions` | int | 5 | Number of 2-mark questions |
| `five_mark_questions` | int | 4 | Number of 5-mark questions |
| `ten_mark_questions` | int | 3 | Number of 10-mark questions |
| `fifteen_mark_questions` | int | 2 | Number of 15-mark questions |
| `easy_percentage` | int | 30 | % of easy questions |
| `medium_percentage` | int | 50 | % of medium questions |
| `hard_percentage` | int | 20 | % of hard questions |
| `institution_name` | str | "University" | Institution name for PDF header |
| `course_name` | str | "Course" | Course name |
| `course_code` | str | "CS101" | Course code |
| `semester` | str | "I" | Semester |
| `exam_type` | str | "End Semester Examination" | Exam type |
| `duration` | str | "3 Hours" | Duration |
| `exam_date` | str | null | Optional exam date |

**Example using curl:**

```bash
curl -X POST http://localhost:8000/generate \
  -F "file=@syllabus.pdf" \
  -F "total_marks=100" \
  -F "two_mark_questions=10" \
  -F "five_mark_questions=4" \
  -F "ten_mark_questions=3" \
  -F "fifteen_mark_questions=2" \
  -F "easy_percentage=30" \
  -F "medium_percentage=50" \
  -F "hard_percentage=20" \
  -F "institution_name=MIT" \
  -F "course_name=Internet of Things" \
  -F "course_code=IOT501" \
  -F "semester=V" \
  -F "exam_type=End Semester Examination"
```

**Response:**

```json
{
  "success": true,
  "message": "Question paper and answer key generated successfully in 42.3s.",
  "final_pdf_path": "generated_papers/question_paper_20240611_143022.pdf",
  "answer_key_pdf_path": "generated_papers/answer_key_20240611_143022.pdf",
  "elapsed_seconds": 42.3,
  "errors": []
}
```

---

### `GET /papers` — List Generated Papers

```bash
curl http://localhost:8000/papers
```

**Response:**

```json
{
  "total": 2,
  "files": [
    "question_paper_20240611_143022.pdf",
    "answer_key_20240611_143022.pdf"
  ]
}
```

---

### `GET /download/{filename}` — Download a PDF

```bash
curl -O http://localhost:8000/download/question_paper_20240611_143022.pdf
```

---

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "llama-3.3-70b-versatile"
}
```

---

## Multi-Agent Workflow

```
START
  │
  ▼
[Syllabus Agent]
  Reads:  uploaded_text
  Output: syllabus_topics (units & topics as JSON)
  │
  ▼
[Question Generator Agent]
  Reads:  syllabus_topics, question_distribution
  Output: generated_questions (id, unit, topic, marks, difficulty)
  │
  ▼
[Bloom Taxonomy Agent]
  Reads:  generated_questions
  Output: bloom_analysis (+ bloom_level, bloom_justification per question)
  │
  ▼
[Validation Agent]
  Reads:  bloom_analysis, syllabus_topics, question_distribution
  Checks: duplicates, coverage, marks distribution, Bloom balance, quality
  Output: validated_questions (corrected if needed)
  │
  ▼
[Answer Key Agent]
  Reads:  validated_questions
  Output: answer_key (model_answer, key_points, marks_breakdown)
  │
  ▼
[PDF Generator]
  Output: question_paper_*.pdf + answer_key_*.pdf
  │
  ▼
END
```

If any agent fails, the workflow stops immediately via LangGraph conditional edges.

---

## Marks Distribution Constraint

The values you pass must satisfy:

```
(two_mark_questions × 2) + (five_mark_questions × 5) +
(ten_mark_questions × 10) + (fifteen_mark_questions × 15) == total_marks

easy_percentage + medium_percentage + hard_percentage == 100
```

The Orchestrator validates this before invoking the workflow.

---

## Logs

All logs are written to `logs/application.log` with rotating file handler (10 MB max, 5 backups).

Console output is color-coded by level:

| Level | Color |
|---|---|
| DEBUG | Cyan |
| INFO | Green |
| WARNING | Yellow |
| ERROR | Red |
| CRITICAL | Magenta |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Future Enhancements

- RAG-based syllabus retrieval using vector databases
- MCQ generation support
- Multi-language question papers
- Faculty review workflow
- Adaptive difficulty based on student performance data
- Hybrid RAG with multiple document sources
