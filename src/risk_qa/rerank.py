"""Cross-encoder inference only on bounded, length-validated candidates."""

import json
from pathlib import Path

from risk_qa.contracts import EvidencePassage
from risk_qa.tokenization import TokenLimits


class Reranker:
    def __init__(self, model, limits):
        self.model, self.limits = model, limits

    @classmethod
    def load(cls, manifest_path: Path, cache_dir: Path):
        from sentence_transformers import CrossEncoder

        spec = json.loads(manifest_path.read_text())["reranker"]
        model = CrossEncoder(spec["id"], revision=spec["revision"], cache_folder=str(cache_dir),
                             local_files_only=True, trust_remote_code=False, device="cpu")
        return cls(model, TokenLimits(manifest_path, cache_dir))

    def rank(self, question: str, candidates: list[EvidencePassage],
             limit: int) -> list[EvidencePassage]:
        candidates = candidates[:30]
        if not candidates:
            return []
        for p in candidates:
            self.limits.validate_pair(question, p.text)
        scores = self.model.predict([(question, p.text) for p in candidates],
                                     batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), x[0].passage_id))
        return [p.model_copy(update={"score": float(s), "ranks": {**p.ranks, "rrf": p.score}})
                for p, s in ranked[:limit]]
