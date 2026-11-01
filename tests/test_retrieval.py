import pytest

from knowledge_agent.rag import KnowledgeBase

CASES = [
    ("年假有几天？", "leave-policy"),
    ("出差住宿标准是多少？", "expense-policy"),
    ("P0 故障多久响应？", "oncall"),
    ("怎么创建 API Key？", "product-faq"),
]


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_expected_document_is_retrieved(kb: KnowledgeBase, query: str, expected: str) -> None:
    hits = kb.search(query, top_k=3)
    assert hits
    assert expected in {hit.chunk.doc_id for hit in hits}


def test_scores_are_sorted(kb: KnowledgeBase) -> None:
    hits = kb.search("报销到账时间", top_k=5)
    assert hits == sorted(hits, key=lambda hit: hit.score, reverse=True)
