from typing import List, Tuple
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents.base import Document

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from src.utils.config import TOP_K
from src.utils.logger import logger


class HybridRetriever:
    """
    Combines dense (FAISS semantic) and sparse (BM25 keyword) retrieval
    using Reciprocal Rank Fusion for final ranking.
    """

    def __init__(self, vectorstore: FAISS, all_chunks: List[Document]):
        self.vectorstore = vectorstore
        self.all_chunks = all_chunks
        # Build BM25 index
        tokenized_corpus = [doc.page_content.lower().split() for doc in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"HybridRetriever initialized with {len(all_chunks)} chunks")

    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results  # List of (Document, score)

    def _sparse_retrieve(self, query: str, k: int) -> List[Document]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.all_chunks[i] for i in top_indices]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Document],
        k: int = 60,
    ) -> List[Document]:
        scores = {}

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, {"doc": doc, "score": 0})
            scores[key]["score"] += 1 / (k + rank + 1)

        for rank, doc in enumerate(sparse_results):
            key = doc.page_content[:100]
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0}
            scores[key]["score"] += 1 / (k + rank + 1)

        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        dense = self._dense_retrieve(query, k=top_k)
        sparse = self._sparse_retrieve(query, k=top_k)
        fused = self._reciprocal_rank_fusion(dense, sparse)
        logger.info(f"Retrieved {len(fused[:top_k])} docs for query: '{query[:60]}'")
        return fused[:top_k]