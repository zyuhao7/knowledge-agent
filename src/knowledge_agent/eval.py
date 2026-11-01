from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent
from .models import Answer
from .rag import KnowledgeBase


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_doc: str | None = None
    expected_terms: list[str] = field(default_factory=list)
    expected_tool: str | None = None


@dataclass
class CaseResult:
    case: Case
    answer: Answer
    retrieval_hit: bool | None
    answer_hit: bool | None
    term_coverage: float
    tool_hit: bool
    passed: bool


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                id=raw["id"],
                question=raw["question"],
                expected_doc=raw.get("expected_doc"),
                expected_terms=raw.get("expected_terms", []),
                expected_tool=raw.get("expected_tool"),
            )
        )
    return cases


def _doc_id(citation: str) -> str:
    return citation.split("#", 1)[0]


def run_cases(
    cases: list[Case], kb: KnowledgeBase, agent: Agent, top_k: int = 5
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        hits = kb.search(case.question, top_k=top_k)
        retrieved = {hit.chunk.doc_id for hit in hits}
        answer = agent.ask(case.question)
        cited = {_doc_id(citation) for citation in answer.citations}

        retrieval_hit = (case.expected_doc in retrieved) if case.expected_doc else None
        answer_hit = (case.expected_doc in cited) if case.expected_doc else None
        coverage = (
            sum(term in answer.text for term in case.expected_terms) / len(case.expected_terms)
            if case.expected_terms
            else 1.0
        )
        tool_hit = (case.expected_tool in answer.tool_calls) if case.expected_tool else True
        passed = (
            coverage >= 0.6
            and tool_hit
            and (answer_hit is not False)
            and (retrieval_hit is not False)
        )
        results.append(
            CaseResult(
                case=case,
                answer=answer,
                retrieval_hit=retrieval_hit,
                answer_hit=answer_hit,
                term_coverage=coverage,
                tool_hit=tool_hit,
                passed=passed,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    return {
        "cases": total,
        "passed": sum(result.passed for result in results),
        "pass_rate": round(sum(result.passed for result in results) / total, 3) if total else 0.0,
        "retrieval_hit_rate": round(
            sum(result.retrieval_hit is True for result in results)
            / max(1, sum(result.retrieval_hit is not None for result in results)),
            3,
        ),
        "answer_hit_rate": round(
            sum(result.answer_hit is True for result in results)
            / max(1, sum(result.answer_hit is not None for result in results)),
            3,
        ),
        "avg_term_coverage": round(sum(result.term_coverage for result in results) / total, 3)
        if total
        else 0.0,
    }


def format_table(results: list[CaseResult]) -> str:
    header = (
        f"{'#':<3} {'retrieval':<10} {'answer':<8} {'terms':<6} {'tool':<6} {'PASS':<5} question"
    )
    lines = [header, "-" * len(header)]
    for index, result in enumerate(results, start=1):
        retrieval = (
            "-" if result.retrieval_hit is None else ("hit" if result.retrieval_hit else "miss")
        )
        answer = "-" if result.answer_hit is None else ("hit" if result.answer_hit else "miss")
        lines.append(
            f"{index:<3} {retrieval:<10} {answer:<8} {result.term_coverage:<6.2f} "
            f"{'ok' if result.tool_hit else 'miss':<6} {'PASS' if result.passed else 'FAIL':<5} "
            f"{result.case.question}"
        )
    return "\n".join(lines)
