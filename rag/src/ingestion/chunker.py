from typing import List
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents.base import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Minimal fallback splitter if langchain's splitter is unavailable
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=512, chunk_overlap=50, separators=None, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n\n", "\n\n", "\n"]
            self.length_function = length_function

        def split_documents(self, docs):
            chunks = []
            for doc in docs:
                text = getattr(doc, 'page_content', str(doc))
                start = 0
                L = len(text)
                while start < L:
                    end = min(start + self.chunk_size, L)
                    chunk_text = text[start:end]
                    # create a new Document with same metadata
                    try:
                        new_doc = Document(page_content=chunk_text, metadata=dict(getattr(doc, 'metadata', {}) or {}))
                    except Exception:
                        # Fallback if Document signature differs
                        new_doc = Document(chunk_text)
                        if hasattr(new_doc, 'metadata'):
                            new_doc.metadata.update(getattr(doc, 'metadata', {}) or {})
                    chunks.append(new_doc)
                    start = end - self.chunk_overlap if end - self.chunk_overlap > start else end

            return chunks
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import logger


# Separators ordered from largest semantic unit to smallest
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Enrich metadata for downstream use
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        # Preserve topic hints if present in content
        lines = chunk.page_content.strip().split("\n")
        if lines:
            chunk.metadata["first_line"] = lines[0][:120]

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks