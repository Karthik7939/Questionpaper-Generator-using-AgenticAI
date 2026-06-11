"""Context-building utilities for retrieved document chunks (no LLM)."""

from typing import Any, Dict, List
import re

from langchain_core.documents.base import Document

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "which", "who", "why", "with", "what's", "whats",
}


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9.]+", text.lower()) if token not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _score_sentence(query_tokens: List[str], sentence: str) -> float:
    sentence_tokens = _tokenize(sentence)
    if not sentence_tokens:
        return 0.0

    overlap = len(set(query_tokens) & set(sentence_tokens))
    score = float(overlap)

    query_numbers = {token for token in query_tokens if re.fullmatch(r"\d+(?:\.\d+)?", token)}
    sentence_numbers = {token for token in sentence_tokens if re.fullmatch(r"\d+(?:\.\d+)?", token)}
    score += 2.0 * len(query_numbers & sentence_numbers)

    if any(token in sentence_tokens for token in ("unit", "module", "topic", "syllabus", "chapter")):
        score += 0.5

    return score


def select_evidence(query: str, context_docs: List[Document], max_sentences: int = 3) -> List[str]:
    """Pick the most query-relevant sentences from retrieved chunks."""
    query_tokens = _tokenize(query)
    scored = []

    for doc in context_docs:
        for sentence in _split_sentences(doc.page_content):
            score = _score_sentence(query_tokens, sentence)
            if score > 0:
                scored.append((score, sentence, doc))

        if not _split_sentences(doc.page_content):
            score = _score_sentence(query_tokens, doc.page_content)
            if score > 0:
                scored.append((score, doc.page_content.strip(), doc))

    if not scored:
        return [doc.page_content[:500] for doc in context_docs[:max_sentences]]

    scored.sort(key=lambda item: item[0], reverse=True)
    evidence = []
    seen = set()
    for _score, sentence, _doc in scored:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        evidence.append(sentence)
        if len(evidence) >= max_sentences:
            break
    return evidence


def document_to_chunk_dict(doc: Document) -> Dict[str, Any]:
    """Serialize a LangChain Document into a plain chunk dict."""
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", "?"),
        "chunk_id": doc.metadata.get("chunk_id", "?"),
        "first_line": doc.metadata.get("first_line", ""),
        "length": doc.metadata.get("chunk_length", len(doc.page_content)),
        "file_type": doc.metadata.get("file_type", ""),
    }


def format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for inclusion in downstream agent prompts."""
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "?")
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[Chunk {chunk_id} | Source: {source}]\n{content}")
    return "\n\n---\n\n".join(parts)
