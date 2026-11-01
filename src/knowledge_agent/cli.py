from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent
from .config import ROOT, Settings
from .embeddings import get_embedder
from .eval import format_table, load_cases, run_cases, summarize
from .llm import LLMClient, LLMError
from .rag import KnowledgeBase
from .tools import ToolRegistry


def build(settings: Settings, use_llm: bool = False) -> tuple[KnowledgeBase, ToolRegistry, Agent]:
    kb = KnowledgeBase(
        settings.data_dir,
        get_embedder(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    kb.ingest()
    tools = ToolRegistry(kb)
    llm = LLMClient(settings) if use_llm else None
    return kb, tools, Agent(kb, tools, llm=llm, top_k=settings.top_k)


def _print_answer(answer) -> None:
    print(answer.text)
    if answer.citations:
        print(f"\n引用：{', '.join(answer.citations)}")
    print(f"模式：{answer.mode}｜工具：{', '.join(answer.tool_calls) or '无'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kagent", description="Knowledge-base agent demo")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_parser = sub.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--llm", action="store_true", help="use the configured chat model")
    ask_parser.add_argument("--json", action="store_true")
    ask_parser.add_argument("--trace", action="store_true")

    search_parser = sub.add_parser("search", help="show retrieval results")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)

    sub.add_parser("docs", help="list ingested documents")
    sub.add_parser("tools", help="list tool schemas")

    eval_parser = sub.add_parser("eval", help="run the evaluation set")
    eval_parser.add_argument("--llm", action="store_true")
    eval_parser.add_argument("--json", action="store_true")
    eval_parser.add_argument("--strict", action="store_true", help="exit 1 when any case fails")

    sub.add_parser("mcp", help="serve tools over MCP stdio")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "mcp":
        from .mcp_server import serve

        serve(settings)
        return 0

    try:
        kb, tools, agent = build(settings, use_llm=getattr(args, "llm", False))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.command == "docs":
        for item in kb.catalog():
            print(f"{item['doc_id']:<16} {item['title']}")
        return 0

    if args.command == "tools":
        print(json.dumps(tools.schemas(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        hits = kb.search(args.query, top_k=args.top_k)
        for rank, hit in enumerate(hits, start=1):
            print(
                f"{rank}. {hit.chunk.id}  score={hit.score:.3f} "
                f"(vector={hit.vector_score:.3f}, keyword={hit.keyword_score:.3f})"
            )
            print(f"   {hit.title}：{hit.chunk.text[:160].replace(chr(10), ' ')}")
        return 0

    if args.command == "ask":
        try:
            answer = agent.ask(args.question)
        except LLMError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(
                json.dumps(
                    {
                        "text": answer.text,
                        "citations": answer.citations,
                        "tool_calls": answer.tool_calls,
                        "mode": answer.mode,
                        "trace": [event.__dict__ for event in answer.trace],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_answer(answer)
            if args.trace:
                for event in answer.trace:
                    print(f"  [{event.step}] {event.detail}")
        return 0

    if args.command == "eval":
        cases = load_cases(ROOT / "evals" / "cases.jsonl")
        results = run_cases(cases, kb, agent, top_k=settings.top_k)
        summary = summarize(results)
        if args.json:
            print(
                json.dumps(
                    {
                        "summary": summary,
                        "cases": [
                            {
                                "id": result.case.id,
                                "passed": result.passed,
                                "retrieval_hit": result.retrieval_hit,
                                "answer_hit": result.answer_hit,
                                "term_coverage": round(result.term_coverage, 3),
                                "tool_hit": result.tool_hit,
                                "answer": result.answer.text,
                                "citations": result.answer.citations,
                            }
                            for result in results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(format_table(results))
            print(
                f"\n通过 {summary['passed']}/{summary['cases']} "
                f"({summary['pass_rate']:.1%})｜检索命中 {summary['retrieval_hit_rate']:.1%}"
                f"｜引用命中 {summary['answer_hit_rate']:.1%}"
                f"｜关键词覆盖 {summary['avg_term_coverage']:.1%}"
            )
        if args.strict and summary["passed"] != summary["cases"]:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
