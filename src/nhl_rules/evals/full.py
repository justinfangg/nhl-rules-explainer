"""Full pipeline evaluation: every case through ask(), scored on routing, retrieval,
verification and citations."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from nhl_rules.config import get_settings
from nhl_rules.evals.cases import EvalCase, rule_matches
from nhl_rules.evals.retrieval import RESULTS_DIR
from nhl_rules.judge import judge
from nhl_rules.llm import Usage
from nhl_rules.pipeline import ask_with_trace
from nhl_rules.retrieval import get_rules

WORKERS = 6


@dataclass
class FullResult:
    id: str
    question: str
    type: str
    style: str
    expected_rules: list[str]
    acceptable_rules: list[str]
    predicted_type: str | None = None
    status: str | None = None  # answered / out_of_scope / unverified, None on error
    retrieved: list[str] = field(default_factory=list)
    first_attempt_passed: bool | None = None  # None if Generate didn't run
    attempts: int = 0
    primary: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    summary: str = ""
    answer: dict | None = None
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
    # Faithfulness judge, for answered cases (None if not judged)
    faithfulness: int | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    judge_rationale: str = ""
    judge_error: str | None = None

    @property
    def in_scope(self) -> bool:
        return self.type != "out_of_scope"

    def hit_at(self, k: int) -> bool:
        return any(rule_matches(r, e) for r in self.retrieved[:k] for e in self.expected_rules)

    @property
    def primary_correct(self) -> bool:
        """The primary citation matches an expected rule."""
        return any(rule_matches(p, e) for p in self.primary for e in self.expected_rules)


def evaluate(case: EvalCase, with_judge: bool = True) -> tuple[FullResult, Usage]:
    r = FullResult(
        case.id, case.question, case.type, case.style, case.expected_rules, case.acceptable_rules
    )
    try:
        answer, usage = ask_with_trace(case.question)
    except Exception as e:  # recorded as a failed case, not a crashed run
        r.error = repr(e)
        return r, Usage()
    d = answer.debug
    r.predicted_type = d.understanding.question_type
    r.status = answer.status
    r.retrieved = [h.number for h in d.retrieved]
    r.attempts = len(d.verification)
    r.first_attempt_passed = d.verification[0].passed if d.verification else None
    if answer.primary:
        r.primary = [c.number for c in answer.primary.rules]
    r.alternatives = [c.number for a in answer.alternatives for c in a.rules]
    r.summary = answer.summary
    r.answer = answer.model_dump(exclude={"debug"})
    r.usage, r.seconds = d.usage, d.seconds

    if with_judge and answer.status == "answered":
        try:
            rules = get_rules([h.number for h in d.retrieved])
            f, judge_usage = judge(case.question, rules, answer)
        except Exception as e:
            r.judge_error = repr(e)
        else:
            r.faithfulness, r.unsupported_claims, r.judge_rationale = (
                f.score,
                f.unsupported_claims,
                f.rationale,
            )
            usage = usage + judge_usage
    return r, usage


def run(cases: list[EvalCase], with_judge: bool = True) -> tuple[list[FullResult], Usage]:
    # Warm the prompt caches with one in-scope case, then run the rest in parallel.
    first = next((c for c in cases if c.type != "out_of_scope"), cases[0])
    rest = [c for c in cases if c is not first]
    outcomes = [evaluate(first, with_judge)]
    if outcomes[0][0].error:  # e.g. bad key or no credit: don't fail every case the same way
        raise RuntimeError(f"first case failed, stopping the run: {outcomes[0][0].error}")
    with ThreadPoolExecutor(WORKERS) as pool:
        outcomes += list(pool.map(lambda c: evaluate(c, with_judge), rest))
    total = Usage()
    for _, u in outcomes:
        total = total + u
    by_id = {r.id: r for r, _ in outcomes}
    return [by_id[c.id] for c in cases], total


def _rate(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def metrics(results: list[FullResult]) -> dict:
    scored = [r for r in results if r.in_scope]
    generated = [r for r in results if r.first_attempt_passed is not None]
    judged = [r.faithfulness for r in results if r.faithfulness is not None]
    return {
        "faithfulness_mean": sum(judged) / len(judged) if judged else None,
        "faithfulness_at_least_4": _rate([f >= 4 for f in judged]) if judged else None,
        "faithfulness_distribution": {s: judged.count(s) for s in range(1, 6)},
        "n_judged": len(judged),
        "routing_accuracy": _rate([r.predicted_type == r.type for r in results]),
        "recall@5": _rate([r.hit_at(5) for r in scored]),
        "recall@8": _rate([r.hit_at(8) for r in scored]),
        # Deterministic check on the first generation, before any retry.
        "verification_pass_rate": _rate([bool(r.first_attempt_passed) for r in generated]),
        "final_verified_rate": _rate([r.status == "answered" for r in generated]),
        # Over all in-scope cases: deflected, unverified or failed cases count as wrong.
        "citation_accuracy": _rate([r.primary_correct for r in scored]),
        "n_cases": len(results),
        "n_generated": len(generated),
        "errors": sum(1 for r in results if r.error),
    }


def write_report(results: list[FullResult], usage: Usage, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s = get_settings()
    m = metrics(results)
    cost = usage.cost(s.llm_model)
    statuses = Counter(r.status or "error" for r in results)
    secs = sorted(r.seconds for r in results if r.status and r.status != "out_of_scope")

    lines = [
        f"# Full eval {stamp}",
        "",
        f"Model `{s.llm_model}` (Understand effort `{s.understand_effort}`, Generate effort "
        f"`{s.generate_effort}`), embedding model `{s.embedding_model}`, hybrid retrieval, "
        f"{m['n_cases']} cases.",
        "",
        "| Metric | Value | Over |",
        "|---|---|---|",
        f"| Routing accuracy | {m['routing_accuracy']:.1%} | all {m['n_cases']} cases |",
        f"| Recall@5 | {m['recall@5']:.1%} | in-scope cases |",
        f"| Recall@8 | {m['recall@8']:.1%} | in-scope cases |",
        f"| Verification pass rate (first try) | {m['verification_pass_rate']:.1%} | "
        f"{m['n_generated']} generated answers |",
        f"| Verified after retry | {m['final_verified_rate']:.1%} | "
        f"{m['n_generated']} generated answers |",
        f"| Citation accuracy (primary) | {m['citation_accuracy']:.1%} | in-scope cases |",
        (
            f"| Faithfulness (judge, 1–5) | mean {m['faithfulness_mean']:.2f}; "
            f"{m['faithfulness_at_least_4']:.0%} scored ≥ 4 | {m['n_judged']} answered cases |"
            if m["n_judged"]
            else "| Faithfulness (judge, 1–5) | not run | |"
        ),
        "",
        "Statuses: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) + ".",
        f"Tokens: {usage.input_tokens:,} input, {usage.cache_read_tokens:,} cache read, "
        f"{usage.cache_write_tokens:,} cache write, {usage.output_tokens:,} output"
        + (f" (≈ ${cost:.2f})." if cost is not None else "."),
    ]
    if m["n_judged"]:
        dist = m["faithfulness_distribution"]
        lines.append(
            "Faithfulness scores: " + ", ".join(f"{k}: {dist[k]}" for k in range(5, 0, -1)) + "."
        )
    if secs:
        lines.append(
            f"Latency for generated answers: median {secs[len(secs) // 2]:.1f}s, "
            f"max {secs[-1]:.1f}s."
        )

    def table(title: str, rows: list[FullResult], cols: list[tuple[str, callable]]) -> None:
        lines.extend(["", f"## {title} ({len(rows)})", ""])
        if not rows:
            return
        lines.append("| Case | " + " | ".join(c for c, _ in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for r in rows:
            lines.append(f"| `{r.id}` | " + " | ".join(str(f(r)) for _, f in cols) + " |")

    table(
        "Errors",
        [r for r in results if r.error],
        [("Error", lambda r: r.error)],
    )
    table(
        "Failed verification on first try",
        [r for r in results if r.first_attempt_passed is False],
        [("Final status", lambda r: r.status), ("Attempts", lambda r: r.attempts)],
    )
    table(
        "Primary citation not an expected rule",
        [r for r in results if r.in_scope and not r.primary_correct],
        [
            ("Expected", lambda r: ", ".join(r.expected_rules)),
            ("Primary", lambda r: ", ".join(r.primary) or r.status),
            ("Retrieved", lambda r: ", ".join(r.retrieved)),
        ],
    )
    table(
        "Faithfulness ≤ 3",
        sorted(
            (r for r in results if r.faithfulness is not None and r.faithfulness <= 3),
            key=lambda r: r.faithfulness,
        ),
        [
            ("Score", lambda r: r.faithfulness),
            ("Unsupported claims", lambda r: " • ".join(r.unsupported_claims).replace("|", "/")),
        ],
    )
    table(
        "Judge errors",
        [r for r in results if r.judge_error],
        [("Error", lambda r: r.judge_error)],
    )
    table(
        "Misrouted",
        [r for r in results if r.predicted_type != r.type],
        [("Labeled", lambda r: r.type), ("Predicted", lambda r: r.predicted_type)],
    )

    md = out_dir / f"{stamp}-full.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / f"{stamp}-full.json").write_text(
        json.dumps(
            {
                "meta": {
                    "timestamp": stamp,
                    "llm_model": s.llm_model,
                    "understand_effort": s.understand_effort,
                    "generate_effort": s.generate_effort,
                    "embedding_model": s.embedding_model,
                    "usage": asdict(usage),
                    "estimated_cost_usd": cost,
                },
                "metrics": m,
                "cases": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return md
