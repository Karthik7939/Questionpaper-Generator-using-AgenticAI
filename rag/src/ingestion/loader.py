import os
from pathlib import Path
from typing import List
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents.base import Document

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
from src.utils.logger import logger


LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".csv": CSVLoader,
}


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        logger.warning(f"Unsupported file type: {ext} for file {file_path}")
        return []
    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        # Attach source metadata
        for doc in docs:
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = ext
        logger.info(f"Loaded {len(docs)} pages/sections from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def load_directory(dir_path: str) -> List[Document]:
    all_docs = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            docs = load_document(full_path)
            all_docs.extend(docs)
    logger.info(f"Total documents loaded from directory: {len(all_docs)}")
    return all_docs