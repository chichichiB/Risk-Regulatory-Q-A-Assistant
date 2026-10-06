from collections import Counter
from pathlib import Path

from risk_qa.eval_contracts import load_dataset, verify_dataset_hash


def test_frozen_dataset_counts_balance_and_separation():
    datasets = {}
    for split, size, per_source in [("benchmark", 50, 8), ("dev", 15, 3)]:
        path = Path(f"eval/{split}.jsonl")
        verify_dataset_hash(path)
        items = load_dataset(path)
        datasets[split] = items
        assert len(items) == size
        counts = Counter(i.primary_doc for i in items if i.answerable)
        assert len(counts) == 5 and set(counts.values()) == {per_source}
        assert not any(i.human_audited for i in items)
    assert {i.question for i in datasets["dev"]}.isdisjoint(i.question for i in datasets["benchmark"])
