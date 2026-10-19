"""Understand-stage evaluation: routing accuracy, and retrieval recall when searching with
the rewritten query instead of the raw question."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from nhl_rules.config import get_settings
from nhl_rules.evals import retrieval as rev
from nhl_rules.evals.cases import EvalCase
from nhl_rules.llm import Usage
from nhl_rules.retrieval import Mode
from nhl_rules.understand import understand_with_usage

TYPES = ("situation", "definition", "judgment", "out_of_scope")
WORKERS = 8


@dataclass
class Routed:
    id: str
    question: str
    expected_type: str
    predicted_type: str | None  # None if the call failed
    rewritten_query: str | None
    explicit_rules: list[str]
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted_type == self.expected_type


def run_understand(cases: list[EvalCase]) -> tuple[list[Routed], Usage]:
    def one(case: EvalCase) -> tuple[Routed, Usage]:
        try:
            u, usage = understand_with_usage(case.question)
        except Exception as e:  # recorded as a failed case, not a crashed run
            return Routed(case.id, case.question, case.type, None, None, [], repr(e)), Usage()
        return (
            Routed(
                case.id,
                case.question,
                case.type,
                u.question_type,
                u.rewritten_query,
                u.explicit_rules,
            ),
            usage,
        )

    # The first call writes the prompt cache; the rest run in parallel and read it.
    outcomes = [one(c) for c in cases[:1]]
    with ThreadPoolExecutor(WORKERS) as pool:
        outcomes += list(pool.map(one, cases[1:]))
    total = Usage()
    for _, usage in outcomes:
        total = total + usage
    return [r for r, _ in outcomes], total


def routing_accuracy(routed: list[Routed]) -> float:
    return sum(r.correct for r in routed) / len(routed) if routed else 0.0


def write_report(
    routed: list[Routed],
    usage: Usage,
    raw: dict[Mode, list[rev.CaseResult]],
    rewritten: dict[Mode, list[rev.CaseResult]],
    out_dir: Path = rev.RESULTS_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = get_settings()
    cost = usage.cost(settings.llm_model)
    failed = [r for r in routed if r.error]

    lines = [
        f"# Understand eval {stamp}",
        "",
        f"Model `{settings.llm_model}` (effort `{settings.understand_effort}`), embedding "
        f"model `{settings.embedding_model}`, {len(routed)} cases.",
        "",
        f"Tokens: {usage.input_tokens:,} input, {usage.cache_read_tokens:,} cache read, "
        f"{usage.cache_write_tokens:,} cache write, {usage.output_tokens:,} output"
        + (f" (≈ ${cost:.2f})." if cost is not None else "."),
        "",
        f"## Routing accuracy: {routing_accuracy(routed):.1%}",
        "",
    ]
    if failed:
        lines += [f"{len(failed)} call(s) failed and count as wrong.", ""]

    confusion = Counter((r.expected_type, r.predicted_type or "error") for r in routed)
    cols = [*TYPES, "error"] if failed else list(TYPES)
    lines += [
        "| Labeled \\ predicted | " + " | ".join(cols) + " | Recall |",
        "|---|" + "---|" * (len(cols) + 1),
    ]
    for t in TYPES:
        n = sum(confusion[(t, c)] for c in cols)
        recall = confusion[(t, t)] / n if n else 0.0
        cells = " | ".join(str(confusion[(t, c)] or "·") for c in cols)
        lines.append(f"| {t} | {cells} | {recall:.0%} |")

    wrong = [r for r in routed if not r.correct]
    lines += ["", f"### Misrouted ({len(wrong)})", ""]
    if wrong:
        lines += ["| Case | Labeled | Predicted | Question |", "|---|---|---|---|"]
    for r in wrong:
        lines.append(
            f"| `{r.id}` | {r.expected_type} | {r.predicted_type or r.error} | {r.question} |"
        )

    lines += [
        "",
        "## Retrieval: raw question vs Understand rewrite",
        "",
        "Rewritten queries also pass any explicit rule numbers to retrieval.",
        "",
        "| Mode | " + " | ".join(f"Raw R@{k} | Rewritten R@{k}" for k in rev.KS) + " |",
        "|---|" + "---|---|" * len(rev.KS),
    ]
    for m in rewritten:
        cells = " | ".join(
            f"{rev.recall(raw[m], k):.0%} | **{rev.recall(rewritten[m], k):.0%}**" for k in rev.KS
        )
        lines.append(f"| {m} | {cells} |")

    groups = rev.summary(next(iter(rewritten.values())))["groups"]
    lines += ["", "### Recall@5 by group (raw → rewritten)", ""]
    lines += [
        "| Group | n | " + " | ".join(rewritten) + " |",
        "|---|---|" + "---|" * len(rewritten),
    ]
    for g, gs in groups.items():
        cells = " | ".join(
            f"{rev.summary(raw[m])['groups'][g]['recall@5']:.0%} → "
            f"{rev.summary(rewritten[m])['groups'][g]['recall@5']:.0%}"
            for m in rewritten
        )
        lines.append(f"| {g} | {gs['n']} | {cells} |")

    for m, rs in rewritten.items():
        misses = [r for r in rs if not r.hit_at(5)]
        lines += ["", f"### Misses at k=5 with rewrite: {m} ({len(misses)})", ""]
        if misses:
            lines += [
                "| Case | Expected | Rewritten query | Retrieved top 5 |",
                "|---|---|---|---|",
            ]
        for r in misses:
            lines.append(
                f"| `{r.id}` | {', '.join(r.expected)} | {r.query} | {', '.join(r.retrieved[:5])} |"
            )

    md = out_dir / f"{stamp}-understand.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / f"{stamp}-understand.json").write_text(
        json.dumps(
            {
                "meta": {
                    "timestamp": stamp,
                    "llm_model": settings.llm_model,
                    "understand_effort": settings.understand_effort,
                    "embedding_model": settings.embedding_model,
                    "usage": asdict(usage),
                    "estimated_cost_usd": cost,
                },
                "routing_accuracy": routing_accuracy(routed),
                "routed": [asdict(r) for r in routed],
                "retrieval": {
                    "raw": {m: rev.summary(rs) for m, rs in raw.items()},
                    "rewritten": {m: rev.summary(rs) for m, rs in rewritten.items()},
                    "rewritten_cases": {m: [asdict(r) for r in rs] for m, rs in rewritten.items()},
                },
            },
            indent=2,
        )
    )
    return md
