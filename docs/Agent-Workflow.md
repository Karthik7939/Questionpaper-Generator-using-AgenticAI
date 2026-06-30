# Multi-Agent Workflow

The project uses a cooperative Multi-Agent architecture where every agent has a single responsibility.

---

## Coordinator (LangGraph)

Responsibilities

- Starts execution
- Passes state between agents
- Handles failures

---

## Syllabus Agent

Input

- Uploaded syllabus

Output

- Units
- Topics
- JSON representation

Responsibilities

- Read syllabus
- Extract academic units
- Remove unnecessary information

---

## Question Generator Agent

Input

- Topics
- Marks distribution

Output

- Questions

Responsibilities

- Generate questions
- Maintain difficulty levels
- Ensure mark distribution

---

## Bloom Agent

Input

- Generated questions

Output

- Bloom level
- Justification

Responsibilities

- Classify every question

---

## Validation Agent

Responsibilities

- Remove duplicates
- Verify syllabus coverage
- Check difficulty balance
- Verify marks distribution

---

## Answer Key Agent

Responsibilities

- Generate model answers
- Generate marking scheme
- Generate key points

---

## Workflow

```
Syllabus

↓

Question Generator

↓

Bloom Classification

↓

Validation

↓

Answer Key

↓

PDF Generation
```