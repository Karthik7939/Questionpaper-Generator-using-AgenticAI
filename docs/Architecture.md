# System Architecture

## Overview

The Agentic AI Question Paper Generator follows a modular architecture based on a Multi-Agent workflow orchestrated using LangGraph. Each agent is responsible for a specific task in the question paper generation pipeline, enabling better modularity, maintainability, and scalability.

---

## Architecture Diagram

```
                +--------------------+
                |   User Uploads     |
                |  Syllabus (PDF/TXT)|
                +----------+---------+
                           |
                           v
                  +------------------+
                  |  FastAPI Server  |
                  +------------------+
                           |
                           v
                 +--------------------+
                 |  LangGraph Workflow|
                 +--------------------+
                           |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v
 Syllabus Agent   Question Generator   Bloom Agent
        |                  |                   |
        +------------------+-------------------+
                           |
                           v
                  Validation Agent
                           |
                           v
                  Answer Key Agent
                           |
                           v
                  PDF Generation Service
                           |
                           v
               Question Paper + Answer Key
```

---

## Core Components

### FastAPI

Acts as the REST API server that receives requests, validates input, and triggers the workflow.

### LangGraph

Coordinates communication between all AI agents using a StateGraph.

### Groq LLM

Provides high-speed inference for all language model tasks.

### PDF Generator

Converts generated content into professional examination papers.

---

## Data Flow

1. User uploads syllabus.
2. Text is extracted.
3. LangGraph starts execution.
4. Each agent processes its assigned task.
5. Final question paper and answer key are generated.
6. PDFs are returned to the user.