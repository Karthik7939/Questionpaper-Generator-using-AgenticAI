# RAG Workflow

## Overview

The Retrieval-Augmented Generation (RAG) pipeline enables the system to generate contextually relevant and syllabus-specific question papers. Instead of relying solely on the Large Language Model (LLM), the RAG pipeline retrieves relevant information from the uploaded syllabus and supplies it as context during question generation.

This approach improves the relevance, accuracy, and consistency of the generated questions while minimizing hallucinations.

---

## Workflow

```text
          Upload Syllabus (PDF/TXT)
                     │
                     ▼
            Document Text Extraction
                     │
                     ▼
             Document Preprocessing
      (Cleaning & Chunk Generation)
                     │
                     ▼
           Generate Text Embeddings
                     │
                     ▼
            Store in Vector Database
                  (FAISS)
                     │
                     ▼
            User Generation Request
                     │
                     ▼
         Convert Query to Embedding
                     │
                     ▼
       Similarity Search (Top-K Chunks)
                     │
                     ▼
          Retrieved Context Package
                     │
                     ▼
             Question Generator Agent
                     │
                     ▼
         LLM Generates Question Paper
```

---

## Workflow Stages

### 1. Document Upload

The user uploads a syllabus document in either PDF or TXT format through the application interface.

Supported formats include:

- PDF
- Plain Text (.txt)

---

### 2. Text Extraction

The uploaded document is parsed to extract readable text.

Responsibilities include:

- Reading PDF pages
- Removing unsupported characters
- Preserving syllabus structure
- Extracting headings and units

Output:

```text
Raw syllabus text
```

---

### 3. Text Preprocessing

Before storing the syllabus, the extracted text undergoes preprocessing.

This includes:

- Removing unnecessary whitespace
- Removing empty lines
- Normalizing text
- Splitting large documents into manageable chunks

Example

Instead of storing:

```text
Unit 1 ...
Unit 2 ...
Unit 3 ...
```

The document is divided into smaller semantic chunks that can be independently retrieved.

---

### 4. Embedding Generation

Each text chunk is converted into a high-dimensional numerical vector using an embedding model.

Embeddings capture the semantic meaning of the syllabus rather than relying on simple keyword matching.

Output:

```text
Chunk
↓

Embedding Vector
```

---

### 5. Vector Database

The generated embeddings are stored inside a FAISS vector database.

Each stored record contains:

- Text chunk
- Embedding vector
- Chunk ID
- Metadata (if available)

This enables efficient similarity searches during question generation.

---

### 6. Query Processing

When the question generation process begins, the system constructs a semantic query based on:

- Marks distribution
- Units
- Difficulty level
- User configuration

The query is converted into an embedding using the same embedding model.

---

### 7. Similarity Search

The query embedding is compared against all stored embeddings using vector similarity search.

The most relevant chunks (Top-K results) are retrieved.

Example

```text
Query

↓

Vector Search

↓

Top 5 Relevant Chunks
```

---

### 8. Context Preparation

The retrieved chunks are combined into a single context package.

This context provides the language model with accurate syllabus information, ensuring that generated questions remain relevant to the uploaded document.

---

### 9. Question Generation

The context package is passed to the Question Generator Agent.

The agent combines:

- Retrieved syllabus context
- Question distribution
- Difficulty constraints
- Prompt templates

to generate the final set of examination questions.

---

## Benefits of Using RAG

- Reduces hallucinations by grounding responses in syllabus content.
- Ensures generated questions align with the uploaded syllabus.
- Improves consistency across multiple question papers.
- Enables scalable retrieval from large documents.
- Supports future expansion to multiple knowledge sources.

---

## Current Implementation

The current RAG module is responsible for:

- Reading uploaded syllabus documents
- Extracting textual content
- Preprocessing documents
- Generating embeddings
- Maintaining the FAISS vector database
- Retrieving relevant syllabus chunks
- Supplying contextual information to the Question Generator Agent

---

## Future Improvements

Potential enhancements to the RAG pipeline include:

- Hybrid Retrieval (Vector + Keyword Search)
- Metadata-based filtering
- Incremental vector database updates
- Multi-document retrieval
- Support for DOCX and PowerPoint files
- Cross-encoder reranking for improved retrieval accuracy
- Context compression for long documents
- Advanced semantic chunking strategies

---

## Summary

The RAG pipeline serves as the knowledge retrieval component of the system. By retrieving relevant syllabus content before invoking the language model, it ensures that generated question papers are accurate, context-aware, and closely aligned with the uploaded academic material.