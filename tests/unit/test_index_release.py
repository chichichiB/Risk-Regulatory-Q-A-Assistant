import pytest

from tests.helpers import passage


def test_sparse_filter_cannot_mix_authority_or_release():
    from risk_qa.sparse import SparseIndex

    index = SparseIndex([passage(), passage("p2", "Operational risk", authority="BCBS")], "r1")
    assert [p.passage_id for p in index.search("operational", "r1", "OSFI", 5)] == ["p1"]
    with pytest.raises(ValueError):
        index.search("operational", "r2", None, 5)


def test_vector_validation_rejects_zero_wrong_dimension_and_nan():
    from risk_qa.embeddings import validate_vector

    for vector in [[0.0] * 384, [1.0] * 12, [float("nan")] * 384]:
        with pytest.raises(ValueError):
            validate_vector(vector)
    assert len(validate_vector([1.0] + [0.0] * 383)) == 384


def test_embedding_query_only_prefix_and_no_truncation():
    from risk_qa.embeddings import Embedder

    class Model:
        def encode(self, texts, **kwargs):
            self.texts = texts
            return [[1.0] + [0.0] * 383 for _ in texts]

    class Limits:
        def validate_query(self, text):
            if len(text) > 128:
                raise ValueError("long query")

        def count(self, text):
            return len(text)

    model = Model()
    embedder = Embedder(model, Limits())
    embedder.encode_query("risk")
    assert model.texts == ["Represent this sentence for searching relevant passages: risk"]
    embedder.encode_passages(["risk"])
    assert model.texts == ["risk"]
    with pytest.raises(ValueError):
        embedder.encode_passages(["x" * 321])
