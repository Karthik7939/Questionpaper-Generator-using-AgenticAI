import os
import pickle
from typing import List

from langchain_core.documents.base import Document

from src.utils.config import EMBEDDING_MODEL, FAISS_INDEX_PATH
from src.utils.logger import logger

_DUMMY_DOCS_FILE = "dummy_docs.pkl"

_HF_LIBS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    _HF_LIBS_AVAILABLE = True
except Exception:
    FAISS = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore


class DummyVectorStore:
    """Token-overlap fallback when HuggingFace embeddings are unavailable."""

    def __init__(self, docs: List[Document] | None = None):
        self.docs = docs or []
        self._token_sets = [set(d.page_content.lower().split()) for d in self.docs]
        self.docstore = type("docstore", (), {"_dict": {i: d for i, d in enumerate(self.docs)}})()

    @classmethod
    def from_documents(cls, docs: List[Document], embedder=None):
        return cls(docs)

    def save_local(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, _DUMMY_DOCS_FILE), "wb") as f:
            pickle.dump(self.docs, f)

    @classmethod
    def load_local(cls, path: str, embedder=None, allow_dangerous_deserialization=False):
        docs_path = os.path.join(path, _DUMMY_DOCS_FILE)
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"No dummy index found at {docs_path}")
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        return cls(docs)

    def add_documents(self, new_docs: List[Document]):
        start = len(self.docs)
        self.docs.extend(new_docs)
        self._token_sets.extend([set(d.page_content.lower().split()) for d in new_docs])
        for i, d in enumerate(new_docs, start=start):
            self.docstore._dict[i] = d

    def similarity_search_with_score(self, query: str, k: int = 5):
        qset = set(query.lower().split())
        scores = []
        for d_set, doc in zip(self._token_sets, self.docs):
            inter = len(qset & d_set)
            score = inter / (1 + len(d_set))
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def _get_embedder():
    if not _HF_LIBS_AVAILABLE or HuggingFaceEmbeddings is None:
        raise ImportError("langchain-huggingface is not installed")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_dummy_index(chunks: List[Document]) -> DummyVectorStore:
    logger.warning(
        "Using token-overlap fallback index (install sentence-transformers for full embeddings)."
    )
    vs = DummyVectorStore.from_documents(chunks)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vs.save_local(FAISS_INDEX_PATH)
    logger.info(f"Dummy index saved to {FAISS_INDEX_PATH} ({len(chunks)} chunks)")
    return vs


def build_faiss_index(chunks: List[Document]):
    logger.info("Building vector index...")
    if _HF_LIBS_AVAILABLE and FAISS is not None:
        try:
            embedder = _get_embedder()
            vectorstore = FAISS.from_documents(chunks, embedder)
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            vectorstore.save_local(FAISS_INDEX_PATH)
            logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
            return vectorstore
        except Exception as exc:
            logger.warning(f"FAISS embedding build failed: {exc}")
    return _build_dummy_index(chunks)


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"No index directory found at {FAISS_INDEX_PATH}. Run ingestion first."
        )

    dummy_docs_path = os.path.join(FAISS_INDEX_PATH, _DUMMY_DOCS_FILE)
    if os.path.exists(dummy_docs_path):
        vs = DummyVectorStore.load_local(FAISS_INDEX_PATH)
        logger.info(f"Dummy index loaded from {FAISS_INDEX_PATH} ({len(vs.docs)} chunks)")
        return vs

    if _HF_LIBS_AVAILABLE and FAISS is not None:
        try:
            embedder = _get_embedder()
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS index loaded from {FAISS_INDEX_PATH}")
            return vectorstore
        except Exception as exc:
            logger.warning(f"FAISS index load failed: {exc}")

    raise FileNotFoundError(f"Index not found or failed to load from {FAISS_INDEX_PATH}")


def update_faiss_index(new_chunks: List[Document]):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    dummy_docs_path = os.path.join(FAISS_INDEX_PATH, _DUMMY_DOCS_FILE)

    if os.path.exists(dummy_docs_path):
        try:
            vs = DummyVectorStore.load_local(FAISS_INDEX_PATH)
            vs.add_documents(new_chunks)
            vs.save_local(FAISS_INDEX_PATH)
            return vs
        except Exception:
            return _build_dummy_index(new_chunks)

    if _HF_LIBS_AVAILABLE and FAISS is not None:
        try:
            embedder = _get_embedder()
            if os.path.exists(FAISS_INDEX_PATH):
                try:
                    vectorstore = FAISS.load_local(
                        FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True
                    )
                    vectorstore.add_documents(new_chunks)
                except Exception:
                    vectorstore = FAISS.from_documents(new_chunks, embedder)
            else:
                vectorstore = FAISS.from_documents(new_chunks, embedder)
            vectorstore.save_local(FAISS_INDEX_PATH)
            return vectorstore
        except Exception as exc:
            logger.warning(f"FAISS index update failed: {exc}")

    return _build_dummy_index(new_chunks)
