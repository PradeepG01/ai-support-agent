"""Knowledge-base retrieval (the "R" in RAG).

Documents are split into overlapping chunks and indexed with TF-IDF (word + bigram).
Queries return the top-k chunks by cosine similarity. The interface is deliberately small so
the TF-IDF index can be swapped for a vector database (pgvector, Qdrant) without touching the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def chunk_markdown(doc_id: str, text: str, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split a markdown document by headings, then window long sections into overlapping chunks."""
    title = doc_id
    first_heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if first_heading:
        title = first_heading.group(1).strip()

    sections = [s.strip() for s in re.split(r"\n(?=##\s)", text) if s.strip()]
    chunks: list[Chunk] = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(Chunk(doc_id, title, section))
            continue
        step = max_words - overlap
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            chunks.append(Chunk(doc_id, title, " ".join(window)))
            if start + max_words >= len(words):
                break
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def documents(self) -> list[str]:
        return sorted({c.doc_id for c in self._chunks})

    def add_document(self, doc_id: str, text: str) -> int:
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]  # re-ingest replaces old version
        new_chunks = chunk_markdown(doc_id, text)
        self._chunks.extend(new_chunks)
        self._reindex()
        return len(new_chunks)

    def load_directory(self, directory: Path) -> int:
        count = 0
        for path in sorted(directory.glob("*.md")):
            count += self.add_document(path.stem, path.read_text(encoding="utf-8"))
        return count

    def _reindex(self) -> None:
        if not self._chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(c.text for c in self._chunks)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[SearchResult]:
        if not self._chunks or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [SearchResult(self._chunks[i], float(s)) for i, s in ranked if s >= min_score]
