"""Explicit, pinned local tokenizers; tests inject counters without model imports."""

import json
from pathlib import Path

from risk_qa.contracts import InputValidationError


class TokenLimits:
    def __init__(self, manifest_path: Path, cache_dir: Path):
        from transformers import AutoTokenizer

        models = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.tokenizers = [AutoTokenizer.from_pretrained(
            entry["id"], revision=entry["revision"], cache_dir=str(cache_dir),
            local_files_only=True, trust_remote_code=False,
        ) for entry in models.values()]

    def count(self, text: str) -> int:
        return max(len(t.encode(text, add_special_tokens=False)) for t in self.tokenizers)

    def validate_query(self, text: str) -> None:
        if self.count(text) > 128:
            raise InputValidationError("Question exceeds 128 model tokens; shorten it")

    def validate_pair(self, question: str, passage: str) -> None:
        self.validate_query(question)
        tok = self.tokenizers[1]
        if len(tok.encode(question, passage, truncation=False)) > min(tok.model_max_length, 512):
            raise InputValidationError("Evidence pair exceeds the reranker input limit")
