import os
import shutil
import time
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document

from src.generation.context_builder import document_to_chunk_dict, select_evidence
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import build_faiss_index, load_faiss_index, update_faiss_index
from src.ingestion.loader import load_document, load_directory
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever
from src.utils.config import FAISS_INDEX_PATH
from src.utils.logger import logger


class RAGPipeline:
    """Chunking, embedding, and retrieval pipeline — no LLM generation."""

    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.all_chunks: List[Document] = []
        self.retriever: Optional[HybridRetriever] = None
        self.reranker = CrossEncoderReranker()

    def reset(self) -> None:
        """Clear in-memory state and the persisted FAISS index."""
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        self.vectorstore = None
        self.all_chunks = []
        self.retriever = None
        logger.info("RAG index reset.")

    def ingest(self, path: str, update: bool = False):
        """Ingest a file or directory into the vector store."""
        if os.path.isdir(path):
            docs = load_directory(path)
        else:
            docs = load_document(path)

        if not docs:
            logger.warning("No documents loaded.")
            return

        chunks = chunk_documents(docs)
        self.all_chunks = chunks

        if update:
            self.vectorstore = update_faiss_index(chunks)
        else:
            self.vectorstore = build_faiss_index(chunks)

        self.retriever = HybridRetriever(self.vectorstore, self.all_chunks)
        logger.info("Ingestion complete. RAG pipeline ready.")

    def ingest_file(self, file_path: str, reset: bool = True) -> None:
        """Ingest a single uploaded file, optionally clearing any prior index."""
        if reset:
            self.reset()
        self.ingest(file_path, update=False)

    def load(self):
        """Load existing FAISS index from disk."""
        self.vectorstore = load_faiss_index()
        self.all_chunks = list(self.vectorstore.docstore._dict.values())
        self.retriever = HybridRetriever(self.vectorstore, self.all_chunks)
        logger.info(f"Loaded index with {len(self.all_chunks)} chunks")

    def get_all_chunks(self) -> List[dict]:
        """Return all ingested chunks as plain dicts."""
        return [document_to_chunk_dict(doc) for doc in self.all_chunks]

    def retrieve(self, query: str, verbose: bool = False) -> dict:
        """Retrieve and rerank relevant chunks for a query (no LLM generation)."""
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load() first.")

        timings = {}
        started_at = time.perf_counter()

        retrieval_started = time.perf_counter()
        retrieved_docs = self.retriever.retrieve(query)
        timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)

        rerank_started = time.perf_counter()
        reranked_docs = self.reranker.rerank(query, retrieved_docs)
        timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)

        chunks = [document_to_chunk_dict(doc) for doc in reranked_docs]
        evidence = select_evidence(query, reranked_docs, max_sentences=5)

        result = {
            "query": query,
            "chunks": chunks,
            "sources": [
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "first_line": chunk["first_line"],
                    "length": chunk["length"],
                }
                for chunk in chunks
            ],
            "debug": {
                "loaded_chunk_count": len(self.all_chunks),
                "retrieved_count": len(retrieved_docs),
                "reranked_count": len(reranked_docs),
                "selected_evidence": evidence,
                "timings_ms": timings,
            },
        }

        if verbose:
            for i, doc in enumerate(reranked_docs):
                logger.debug(f"Context [{i + 1}]: {doc.page_content[:200]}")

        return result
