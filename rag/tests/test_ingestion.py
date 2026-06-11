import pytest
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents

def test_load_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Chapter 1: Introduction\nThis is a test syllabus document.\nTopic 1: Basics\n")
    docs = load_document(str(f))
    assert len(docs) > 0
    assert "Introduction" in docs[0].page_content

def test_chunking():
    try:
        from langchain.schema import Document
    except Exception:
        from langchain_core.documents.base import Document
    docs = [Document(page_content="A " * 600, metadata={"source": "test"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600