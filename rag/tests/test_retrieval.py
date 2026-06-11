import pytest
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents.base import Document
from src.retrieval.retriever import HybridRetriever
from unittest.mock import MagicMock

def make_mock_vectorstore(docs):
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = [(doc, 0.9) for doc in docs]
    return vs

def test_hybrid_retriever():
    docs = [
        Document(page_content="Photosynthesis is the process plants use to make food.", metadata={}),
        Document(page_content="Newton's laws of motion describe forces.", metadata={}),
        Document(page_content="Trigonometry covers sine cosine and tangent.", metadata={}),
    ]
    vs = make_mock_vectorstore(docs)
    retriever = HybridRetriever(vs, docs)
    results = retriever.retrieve("photosynthesis", top_k=2)
    assert len(results) <= 2