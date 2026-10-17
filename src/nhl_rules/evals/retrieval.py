"""Retrieval-only evaluation: recall@k for each search mode over the eval set."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from nhl_rules.config import get_settings
from nhl_rules.evals.cases import EvalCase, rule_matches
from nhl_rules.retrieval import TOP_K, Mode, search

KS = (5, 8)
RESULTS_DIR = Path("eval/results")


@dataclass
class CaseResult:
    id: str
    question: str
    type: str
    style: str
    expected: list[str]
    retrieved: list[str]
    # 1-based rank of the first retrieved rule matching an expected rule, or None
    first_hit: int | None

    def hit_at(self, k: int) -> bool:
        return self.first_hit is not None and self.first_hit <= k


def evaluate_case(case: EvalCase, mode: Mode) -> CaseResult:
    retrieved = [h.rule_number for h in search(case.question, mode, k=max(KS))]
    first_hit = next(
        (
            i
            for i, r in enumerate(retrieved, start=1)
            if any(rule_matches(r, e) for e in case.expected_rules)
        ),
        None,
    )
    return CaseResult(
        case.id, case.question, case.type, case.style, case.expected_rules, retrieved, first_hit
    )


def recall(results: list[CaseResult], k: int) -> float:
    return sum(r.hit_at(k) for r in results) / len(results) if results else 0.0


def run(cases: list[EvalCase], modes: list[Mode]) -> dict[Mode, list[CaseResult]]:
    """Out-of-scope cases have no expected rules, so they're skipped here."""
    scored = [c for c in cases if c.type != "out_of_scope"]
    return {mode: [evaluate_case(c, mode) for c in scored] for mode in modes}


def summary(results: list[CaseResult]) -> dict:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[f"style={r.style}"].append(r)
        groups[f"type={r.type}"].append(r)
    return {
        "n": len(results),
        **{f"recall@{k}": recall(results, k) for k in KS},
        "groups": {
            g: {"n": len(rs), **{f"recall@{k}": recall(rs, k) for k in KS}}
            for g, rs in sorted(groups.items())
        },
    }


def write_report(by_mode: dict[Mode, list[CaseResult]], out_dir: Path = RESULTS_DIR) -> Path:
    """Write a markdown report and a JSON file with every case's results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = get_settings()
    meta = {
        "timestamp": stamp,
        "embedding_model": settings.embedding_model,
        "season": settings.rulebook_season,
        "top_k": TOP_K,
    }
    summaries = {m: summary(rs) for m, rs in by_mode.items()}

    lines = [
        f"# Retrieval eval {stamp}",
        "",
        f"Embedding model `{meta['embedding_model']}`, season {meta['season']}, "
        f"{next(iter(summaries.values()))['n']} cases (out-of-scope excluded). A case is a hit "
        "if any expected rule, its parent or one of its subsections is in the top k.",
        "",
        "| Mode | " + " | ".join(f"Recall@{k}" for k in KS) + " |",
        "|---|" + "---|" * len(KS),
    ]
    for m, s in summaries.items():
        lines.append(f"| {m} | " + " | ".join(f"{s[f'recall@{k}']:.0%}" for k in KS) + " |")

    groups = next(iter(summaries.values()))["groups"]
    lines += ["", "## Recall@5 by group", "", "| Group | n | " + " | ".join(summaries) + " |"]
    lines.append("|---|---|" + "---|" * len(summaries))
    for g, gs in groups.items():
        cells = " | ".join(f"{summaries[m]['groups'][g]['recall@5']:.0%}" for m in summaries)
        lines.append(f"| {g} | {gs['n']} | {cells} |")

    for m, rs in by_mode.items():
        misses = [r for r in rs if not r.hit_at(5)]
        lines += ["", f"## Misses at k=5: {m} ({len(misses)})", ""]
        if misses:
            lines += ["| Case | Expected | Retrieved top 5 | First hit |", "|---|---|---|---|"]
        for r in misses:
            lines.append(
                f"| `{r.id}` | {', '.join(r.expected)} | {', '.join(r.retrieved[:5])} "
                f"| {r.first_hit or '–'} |"
            )

    md = out_dir / f"{stamp}-retrieval.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / f"{stamp}-retrieval.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summaries,
                "cases": {m: [asdict(r) for r in rs] for m, rs in by_mode.items()},
            },
            indent=2,
        )
    )
    return md
