"""Deterministic sparse ranking over one immutable corpus release."""

import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from risk_qa.contracts import EvidencePassage


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


class SparseIndex:
    def __init__(self, passages: list[EvidencePassage], release_id: str, mode: str = "bm25"):
        if any(p.release_id != release_id for p in passages):
            raise ValueError("Sparse index cannot mix releases")
        self.passages, self.release_id, self.mode = passages, release_id, mode
        self._indexes = {}

    def search(self, query: str, release_id: str, authority: str | None,
               limit: int) -> list[EvidencePassage]:
        if release_id != self.release_id:
            raise ValueError("Sparse release mismatch")
        if authority not in self._indexes:
            ps = [p for p in self.passages if authority is None or p.authority == authority]
            if not ps:
                return []
            if self.mode == "tfidf":
                model = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
                vectors = model.fit_transform([p.text for p in ps])
            else:
                model, vectors = BM25Okapi([tokenize(p.text) for p in ps]), None
            self._indexes[authority] = ps, model, vectors
        ps, model, vectors = self._indexes[authority]
        if self.mode == "tfidf":
            scores = (vectors @ model.transform([query]).T).toarray().ravel()
        else:
            scores = model.get_scores(tokenize(query))
        ranked = sorted(zip(ps, scores), key=lambda x: (-float(x[1]), x[0].passage_id))
        return [p.model_copy(update={"score": float(s)}) for p, s in ranked[:limit]]
