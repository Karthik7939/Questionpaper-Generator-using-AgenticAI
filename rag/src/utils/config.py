import os
from pathlib import Path
from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(RAG_ROOT / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = str(RAG_ROOT / os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K", 8))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 4))
UPLOAD_DIR = str(RAG_ROOT / "data" / "uploads")
