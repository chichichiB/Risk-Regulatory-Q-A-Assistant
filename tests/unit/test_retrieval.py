import pytest

from tests.helpers import passage


def test_rrf_uses_ranks_not_raw_scores_and_deduplicates():
    from risk_qa.retrieval import reciprocal_rank_fusion

    a, b = passage("a"), passage("b")
    found = reciprocal_rank_fusion({"dense": [a, b], "bm25": [b, a]}, "r1")
    assert [p.passage_id for p in found] == ["a", "b"]
    assert found[0].score == pytest.approx(1 / 61 + 1 / 62)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": [passage(release="other")]}, "r1")


def test_retriever_keeps_authority_through_reranking():
    from risk_qa.retrieval import Retriever

    class Store:
        def all_passages(self, release):
            return [passage(), passage("b", authority="BCBS")]

        def search_dense(self, vector, release, authority, limit):
            return [passage()]

    class Embed:
        def encode_query(self, text):
            return [1] + [0] * 383

    class Rank:
        def rank(self, question, candidates, limit):
            return list(reversed(candidates))[:limit]

    r = Retriever(Store(), Embed(), Rank())
    found = r.search("risk", "r1", "OSFI")
    assert found and all(p.authority == "OSFI" for p in found)
    assert len(found) <= 5
