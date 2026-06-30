# Installation Guide

## Requirements

- Python 3.11+
- Groq API Key
- pip

---

## Clone Repository

```bash
git clone <repository_url>

cd question-paper-generator
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create

```
.env
```

Example

```
GROQ_API_KEY=your_key
MODEL_NAME=llama-3.3-70b-versatile
```

---

## Run

```bash
python -m app.main
```

or

```bash
uvicorn app.main:app --reload
```

Open

```
http://localhost:8000/docs
```