from knowledge_agent.agent import Agent
from knowledge_agent.tools import ToolRegistry


def test_offline_answer_cites_source(kb) -> None:
    agent = Agent(kb, ToolRegistry(kb), top_k=5)
    answer = agent.ask("年假有几天？")
    assert "leave-policy" in answer.text
    assert "5 天" in answer.text
    assert answer.tool_calls == ["search_docs"]
    assert any(citation.startswith("leave-policy#") for citation in answer.citations)


def test_offline_routes_math_to_calculator(kb) -> None:
    agent = Agent(kb, ToolRegistry(kb), top_k=5)
    answer = agent.ask("128*3+16")
    assert answer.text.strip() == "128*3+16 = 400"
    assert answer.tool_calls == ["calculate"]


def test_detail_question_reads_full_document(kb) -> None:
    agent = Agent(kb, ToolRegistry(kb), top_k=5)
    answer = agent.ask("请给出差旅住宿标准全文")
    assert "差旅与报销制度" in answer.text
    assert "read_doc" in answer.tool_calls


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_docs",
                            "arguments": '{"query": "年假", "top_k": 3}',
                        },
                    }
                ],
            }
        assert any(message["role"] == "tool" for message in messages)
        return {"role": "assistant", "content": "年假按司龄为 5/10/15 天 [leave-policy#0]。"}


def test_llm_loop_executes_tool_and_extracts_citation(kb) -> None:
    agent = Agent(kb, ToolRegistry(kb), llm=FakeLLM(), top_k=5)
    answer = agent.ask("年假有几天？")
    assert answer.mode == "llm"
    assert answer.tool_calls == ["search_docs"]
    assert "leave-policy#0" in answer.citations
    assert "5/10/15" in answer.text
