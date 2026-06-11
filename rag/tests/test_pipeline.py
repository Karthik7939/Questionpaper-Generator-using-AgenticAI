import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.rag_pipeline import RAGPipeline


def test_pipeline_load_and_retrieve():
    pipeline = RAGPipeline()
    with patch.object(pipeline, 'load'), \
         patch.object(pipeline, 'retriever') as mock_ret, \
         patch.object(pipeline, 'reranker') as mock_rerank:

        try:
            from langchain.schema import Document
        except Exception:
            from langchain_core.documents.base import Document
        mock_doc = Document(
            page_content="Test content about syllabus units and topics",
            metadata={"source": "test", "chunk_id": 0, "first_line": "Test", "chunk_length": 12},
        )
        mock_ret.retrieve.return_value = [mock_doc]
        mock_rerank.rerank.return_value = [mock_doc]
        pipeline.retriever = mock_ret
        pipeline.reranker = mock_rerank

        result = pipeline.retrieve("syllabus units topics")
        assert "chunks" in result
        assert "sources" in result
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["content"] == "Test content about syllabus units and topics"
