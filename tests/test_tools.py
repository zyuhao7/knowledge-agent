import pytest

from knowledge_agent.tools import ToolRegistry, safe_calculate


def test_calculator_handles_precedence() -> None:
    assert safe_calculate("2+3*4") == 14
    assert safe_calculate("(2+3)*4") == 20
    assert safe_calculate("-2**3") == -8


@pytest.mark.parametrize("expression", ["__import__('os')", "a+1", "2**100", "[1,2]"])
def test_calculator_rejects_unsafe_input(expression: str) -> None:
    with pytest.raises(ValueError):
        safe_calculate(expression)


def test_dispatch_search_returns_citations(tools: ToolRegistry) -> None:
    result = tools.dispatch("search_docs", {"query": "年假", "top_k": 3})
    assert result.ok
    assert "leave-policy#" in result.content
    assert result.meta["doc_ids"]


def test_dispatch_unknown_tool_is_an_error(tools: ToolRegistry) -> None:
    result = tools.dispatch("does_not_exist", {})
    assert not result.ok
    assert "unknown tool" in result.content


def test_read_doc_returns_full_text(tools: ToolRegistry) -> None:
    result = tools.dispatch("read_doc", {"doc_id": "expense-policy"})
    assert result.ok
    assert "住宿标准" in result.content
