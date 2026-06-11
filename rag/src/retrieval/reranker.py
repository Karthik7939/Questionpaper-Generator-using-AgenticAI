from typing import List
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents.base import Document
from src.utils.config import RERANK_TOP_N
from src.utils.logger import logger


try:
    from sentence_transformers import CrossEncoder


    class CrossEncoderReranker:
        """Uses a CrossEncoder for reranking when available."""

        def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            self.model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder reranker loaded: {model_name}")

        def rerank(self, query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
            if not docs:
                return []
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self.model.predict(pairs)
            scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in scored_docs[:top_n]]
            logger.info(f"Reranked to top {top_n} docs. Best score: {scored_docs[0][0]:.4f}")
            return top_docs
except Exception:
    # Fallback: no-op reranker
    class CrossEncoderReranker:
        def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            logger.warning("CrossEncoder not available; using no-op reranker.")

        def rerank(self, query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
            return docs[:top_n]