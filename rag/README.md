# Exam RAG — Project Documentation

**Purpose**
- This project is a Retrieval-Augmented Generation (RAG) pipeline designed to ingest documents, build a searchable vector index, retrieve relevant context for user queries, rerank results, and generate answers using a local LLM (Ollama) or a fallback. It is tailored for exam question/answer workflows but is general-purpose for RAG use.

**Quick Start**
- Create and activate your Python virtual environment and install dependencies (in the project root):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt    # or at least: flask httpx faiss-cpu langchain langchain-community
pip install flask
```

- Start the web UI (development):

```
py webapp/app.py
```

- Open http://localhost:8000, upload documents and ask questions. Uploaded files are saved to the `data/uploads` folder and ingested into the index.

**Project Structure & Key Files**
- **Entry / CLI**: [main.py](main.py) — interactive CLI wrapper used during development to ingest or load and query the pipeline.
- **Web UI**: [webapp/app.py](webapp/app.py) — Flask app exposing `/` (UI), `/upload` (POST file upload & ingest), and `/query` (POST question → JSON answer). Static UI is in [webapp/templates/index.html](webapp/templates/index.html) and [webapp/static/style.css](webapp/static/style.css).
- **Pipeline orchestrator**: [src/pipeline/rag_pipeline.py](src/pipeline/rag_pipeline.py) — `RAGPipeline` class; methods:
  - `ingest(path, update=False)` — load files, chunk, and build/update index.
  - `load()` — load an existing index from disk and reconstruct chunks.
  - `query(question)` — run retrieval → rerank → generate, returning JSON-style result.
- **Ingestion**:
  - [src/ingestion/loader.py](src/ingestion/loader.py) — document loaders for many file types (pdf, pptx, txt) using community loaders; returns `Document` objects.
  - [src/ingestion/chunker.py](src/ingestion/chunker.py) — text chunking logic (uses `RecursiveCharacterTextSplitter` if available, with `CHUNK_SIZE` and `CHUNK_OVERLAP`).
  - [src/ingestion/embedder.py](src/ingestion/embedder.py) — builds/loads/updates FAISS vectorstore. Uses `langchain_huggingface.HuggingFaceEmbeddings` when available; otherwise a lightweight `DummyVectorStore` fallback is used so the pipeline remains testable without heavy HF dependencies.
- **Retrieval & Rerank**:
  - [src/retrieval/retriever.py](src/retrieval/retriever.py) — `HybridRetriever` combining dense FAISS similarity and sparse BM25 ranking; merges results with Reciprocal Rank Fusion.
  - [src/retrieval/reranker.py](src/retrieval/reranker.py) — optional `CrossEncoderReranker` using `sentence_transformers` when available; falls back to a no-op top-N slice when not installed.
- **Generation**: [src/generation/generator.py](src/generation/generator.py) — builds prompts from retrieved evidence and calls Ollama HTTP API to generate answers. The code now selects and sends only the top evidence sentences (scored by overlap and numeric matches) to reduce hallucination and uses a low temperature for deterministic answers. If Ollama is unavailable, generation will return an error string.
- **Utilities**:
  - [src/utils/config.py](src/utils/config.py) — configuration values and environment variable mappings: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `FAISS_INDEX_PATH`, `UPLOAD_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RERANK_TOP_N`.
  - [src/utils/logger.py](src/utils/logger.py) — logging helper used across modules.

**Storage & Files**
- Uploaded documents: `data/uploads/` — files saved by the web UI.
- Vector index: `vectorstore/faiss_index/` — directory where FAISS saves index files. The project ensures directories exist before saving; if the index cannot be read, ingestion rebuilds it from the current uploads.

**How the system works (detailed flow)**
1. Upload / Ingest
   - Uploads via the web UI (`/upload`) save files to `data/uploads`.
   - The upload handler resets any existing FAISS index (so deleted files stop appearing) and then calls `RAGPipeline.ingest()` on the `data/uploads` folder to rebuild the index from all files.

2. Loading documents
   - `loader.py` detects file types and converts them into `Document` objects with `page_content` and metadata such as `source` and `page`.

3. Chunking
   - `chunker.py` splits long documents into overlapping chunks; metadata added to each chunk includes `chunk_id`, `chunk_length`, and `first_line` for easy reference.

4. Embedding & Indexing
   - `embedder.py` constructs embeddings using `HuggingFaceEmbeddings` when possible and builds a FAISS vectorstore with `FAISS.from_documents()`.
   - If heavy HF deps are missing or fail to import, a `DummyVectorStore` fallback stores token-overlap info and allows testing retrieval.
   - Index files are persisted under `vectorstore/faiss_index/`.

5. Retrieval
   - `HybridRetriever` runs both dense similarity (`FAISS.similarity_search_with_score`) and sparse BM25 scoring, then fuses them via reciprocal rank fusion.

6. Reranking
   - If `sentence_transformers.CrossEncoder` is available the top retrieved docs are reranked by the cross-encoder; otherwise a simple top-N pass is used.

7. Answer Generation
   - The system selects the most relevant sentences from the retrieved docs (evidence selector that scores sentence overlap, numeric matches, and domain keywords).
   - Only the top evidence snippets are included in the prompt to the LLM (Ollama) to reduce hallucination.
   - The LLM produces the answer which is returned along with source metadata used for citation.

**Configuration & Environment**
- Environment variables (supported in `.env`):
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
   - `OLLAMA_MODEL` (default: `qwen3:4b`)
  - `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
  - `FAISS_INDEX_PATH` (default: `vectorstore/faiss_index`)
  - `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RERANK_TOP_N`

**Common Issues & Troubleshooting**
- FAISS file I/O errors ("could not open index.faiss"): remove the `vectorstore/faiss_index/` directory and re-run ingestion. The web UI upload flow will rebuild the index from `data/uploads`.

- Sentence-Transformers / HuggingFace compatibility errors: these libraries are version-sensitive. Options:
  - Install pinned compatible versions in your active venv, e.g. `pip install "huggingface-hub==0.13.4" "sentence-transformers==2.2.2"` (or experiment with other compatible pairs).
  - Continue using the `DummyVectorStore` fallback for development/testing without heavy HF dependencies.

- Stale results after deleting files: ensure you remove the files from `data/uploads` and either restart the web app or use the web UI upload (which triggers a rebuild). The web UI also resets index state on upload to ensure deleted files are not retained.

**Extending the system**
- Add a persistent metadata store for chunk <-> original-file mapping (e.g., SQLite) to enable precise delete-by-file operations without full index rebuilds.
- Add more sophisticated sentence-extraction and numeric-parsing for factual Q/A (dates, currency, units).
- Integrate a hosted embedding API (OpenAI or others) if you need reliable, low-maintenance embeddings instead of local HF models.

**Files added by this documentation task**
- README: [README.md](README.md)

If you want, I can commit this README into the repo (done) and also generate a compact `CONTRIBUTING.md` and `ARCHITECTURE.md` with diagrams. What would you like next?
