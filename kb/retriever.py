from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .tokenize import tokenize


@dataclass
class Hit:
    score: float
    text: str
    meta: dict
    chunk_id: str


class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._corpus_tokens = [tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        try:
            best = float(max(scores)) if len(scores) else 0.0
        except Exception:
            best = 0.0
        # If nothing matches (common for cross-lingual queries), don't return arbitrary documents.
        if best <= 0.0:
            return []
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, top_k)]
        hits: list[dict] = []
        for i in idxs:
            c = self._chunks[i]
            hits.append(
                {
                    "score": float(scores[i]),
                    "id": c.get("id", str(i)),
                    "text": c.get("text", ""),
                    "meta": c.get("meta", {}),
                }
            )
        return hits
