"""Rank fusion and reranking with corpus and authority invariants."""

from risk_qa.contracts import EvidencePassage
from risk_qa.sparse import SparseIndex, tokenize


def reciprocal_rank_fusion(rankings: dict[str, list[EvidencePassage]],
                           release_id: str) -> list[EvidencePassage]:
    found: dict[str, EvidencePassage] = {}
    scores: dict[str, float] = {}
    traces: dict[str, dict[str, float]] = {}
    for channel, ranked in rankings.items():
        seen = set()
        for rank, p in enumerate(ranked, 1):
            if p.release_id != release_id:
                raise ValueError("Candidate belongs to another corpus release")
            if p.passage_id in seen:
                continue
            seen.add(p.passage_id)
            found[p.passage_id] = p
            scores[p.passage_id] = scores.get(p.passage_id, 0) + 1 / (60 + rank)
            traces.setdefault(p.passage_id, {})[channel] = rank
    return [found[pid].model_copy(update={"score": scores[pid], "ranks": traces[pid]})
            for pid in sorted(found, key=lambda x: (-scores[x], x))]


class Retriever:
    def __init__(self, store, embedder, reranker):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self._sparse = {}

    def search(self, question: str, release_id: str, authority: str | None,
               limit: int = 5, mode: str = "reranked") -> list[EvidencePassage]:
        if mode not in {"dense", "bm25", "tfidf", "hybrid", "reranked"}:
            raise ValueError("Unknown retrieval mode")
        limit = min(max(limit, 1), 5)
        key = (release_id, "tfidf" if mode == "tfidf" else "bm25")
        if key not in self._sparse:
            self._sparse[key] = SparseIndex(self.store.all_passages(release_id), *key)
        sparse = self._sparse[key].search(question, release_id, authority, 20)
        if mode in {"bm25", "tfidf"}:
            return sparse[:limit]
        dense = self.store.search_dense(self.embedder.encode_query(question), release_id, authority, 20)
        if any(p.authority != authority for p in dense if authority is not None):
            raise ValueError("Dense retrieval ignored its authority filter")
        if mode == "dense":
            return dense[:limit]
        rankings = {"dense": dense, "bm25": sparse}
        identifiers = {t for t in tokenize(question) if any(ch.isdigit() for ch in t)}
        if identifiers:
            exact = [p for p in self._sparse[key].passages
                     if (authority is None or p.authority == authority)
                     and identifiers.intersection(tokenize(p.doc_id + " " + p.text))]
            if exact:
                rankings["identifier"] = sorted(exact, key=lambda p: p.passage_id)[:20]
        fused = reciprocal_rank_fusion(rankings, release_id)
        if mode == "hybrid":
            return fused[:limit]
        candidates = fused[:30]
        result = self.reranker.rank(question, candidates, limit)
        allowed = {p.passage_id: p for p in candidates}
        if any(p.passage_id not in allowed or p.release_id != release_id
               or (authority is not None and p.authority != authority) for p in result):
            raise ValueError("Reranker returned foreign evidence")
        return result[:limit]
