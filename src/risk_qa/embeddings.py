"""Pinned CPU bi-encoder with explicit token and vector validation."""

import json
import math
from pathlib import Path

from risk_qa.tokenization import TokenLimits

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def validate_vector(vector) -> list[float]:
    values = [float(x) for x in vector]
    if len(values) != 384 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected 384 finite embedding dimensions")
    if sum(x * x for x in values) == 0:
        raise ValueError("A zero vector has no cosine similarity")
    return values


class Embedder:
    def __init__(self, model, limits):
        self.model, self.limits = model, limits

    @classmethod
    def load(cls, manifest_path: Path, cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        spec = json.loads(manifest_path.read_text())["embedding"]
        model = SentenceTransformer(spec["id"], revision=spec["revision"],
                                    cache_folder=str(cache_dir), local_files_only=True,
                                    trust_remote_code=False, device="cpu")
        return cls(model, TokenLimits(manifest_path, cache_dir))

    def encode_query(self, text: str) -> list[float]:
        self.limits.validate_query(text)
        return validate_vector(self.model.encode([QUERY_PREFIX + text],
                                                  normalize_embeddings=True)[0])

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if any(self.limits.count(t) > 320 for t in texts):
            raise ValueError("Passage exceeds the frozen 320-token limit")
        return [validate_vector(v) for v in self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)]
