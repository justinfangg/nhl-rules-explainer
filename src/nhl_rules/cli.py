"""`nhl` command-line client. A thin wrapper over the HTTP API."""

import re
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer

from nhl_rules import glossary as g
from nhl_rules.config import get_settings
from nhl_rules.db import apply_schema

app = typer.Typer(help="NHL Rules Explainer CLI.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """NHL Rules Explainer CLI."""


@app.command()
def health() -> None:
    """Check that the API, database and models are ready."""
    url = get_settings().nhl_api_url
    try:
        resp = httpx.get(f"{url}/health", timeout=10)
    except httpx.HTTPError as e:
        typer.secho(f"API unreachable at {url}: {e}", fg="red", err=True)
        raise typer.Exit(1) from None
    if resp.status_code not in (200, 503):
        typer.secho(
            f"Unexpected response from {url}/health: {resp.status_code}", fg="red", err=True
        )
        raise typer.Exit(1)

    body = resp.json()
    for name, check in body["checks"].items():
        mark, color = ("✓", "green") if check["ok"] else ("✗", "red")
        typer.secho(f"{mark} {name}: {check['detail']}", fg=color)
    healthy = body["status"] == "ok"
    typer.secho("healthy" if healthy else "unhealthy", fg="green" if healthy else "red", bold=True)
    raise typer.Exit(0 if healthy else 1)


@app.command()
def ingest(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Rulebook PDF")],
    season: Annotated[str, typer.Option(help='Season label, e.g. "2026-27"')],
) -> None:
    """Parse the rulebook PDF and load it into the database (runs locally, not via the API)."""
    # Imported here so other commands don't pay for loading PyMuPDF and torch.
    from nhl_rules.ingest.loader import load
    from nhl_rules.ingest.parser import parse_pdf

    if not re.fullmatch(r"\d{4}-\d{2}", season):
        raise typer.BadParameter('expected a season like "2026-27"', param_hint="--season")

    model = get_settings().embedding_model
    start = time.monotonic()
    chunks = parse_pdf(pdf)
    rules = sum(1 for c in chunks if c.parent_number is None and c.section != "Glossary of Terms")
    subsections = sum(1 for c in chunks if c.parent_number)
    definitions = sum(1 for c in chunks if c.section == "Glossary of Terms")
    typer.echo(f"Parsed {rules} rules, {subsections} subsections, {definitions} definitions.")
    typer.echo(f"Embedding with {model} and loading season {season}...")
    n = load(chunks, season, model)
    typer.secho(f"Loaded {n} rows in {time.monotonic() - start:.0f}s.", fg="green")
    if g.DEFAULT_PATH.exists():
        glossary(g.DEFAULT_PATH, season)


@app.command()
def glossary(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = g.DEFAULT_PATH,
    season: Annotated[str | None, typer.Option(help="Season to check rule numbers against")] = None,
) -> None:
    """Validate the fan-term glossary against the loaded rules and load it (runs locally)."""
    season = season or get_settings().rulebook_season
    terms = g.read_terms(path)
    apply_schema()
    missing = g.missing_rules(terms, season)
    if missing:
        for term, nums in missing.items():
            typer.secho(f"✗ {term!r}: no rule {', '.join(nums)} in season {season}", fg="red")
        typer.secho(f"Glossary not loaded: {len(missing)} term(s) cite unknown rules.", fg="red")
        raise typer.Exit(1)
    n = g.load(terms)
    typer.secho(f"Loaded {n} glossary terms; all rule numbers exist in {season}.", fg="green")


@app.command()
def rule(number: Annotated[str, typer.Argument(help='Rule number, e.g. "56.1" or "56"')]) -> None:
    """Print one rule (with its parent rule's title), or all subsections of a rule."""
    url = get_settings().nhl_api_url
    try:
        resp = httpx.get(f"{url}/rules/{number}", timeout=10)
    except httpx.HTTPError as e:
        typer.secho(f"API unreachable at {url}: {e}", fg="red", err=True)
        raise typer.Exit(1) from None
    if resp.status_code == 404:
        typer.secho(f"No rule {number}.", fg="red", err=True)
        raise typer.Exit(1)
    resp.raise_for_status()
    body = resp.json()
    for r in body["subsections"] or [body["rule"]]:
        parent = (
            f"Rule {r['parent']['rule_number']} – {r['parent']['title']} › " if r["parent"] else ""
        )
        typer.secho(f"\n{parent}{r['rule_number']} {r['title']}", bold=True)
        typer.secho(r["section"], dim=True)
        typer.echo(r["text"])


@app.command("eval")
def eval_(
    retrieval_only: Annotated[
        bool, typer.Option("--retrieval-only", help="Measure retrieval recall only")
    ] = False,
    mode: Annotated[
        str | None, typer.Option(help="keyword, vector or hybrid (default: all three)")
    ] = None,
    cases_path: Annotated[Path, typer.Option("--cases")] = Path("eval/cases.json"),
) -> None:
    """Run the eval set and write a report to eval/results/ (runs locally)."""
    from nhl_rules.evals import retrieval as rev
    from nhl_rules.evals.cases import read_cases
    from nhl_rules.retrieval import MODES

    if not retrieval_only:
        typer.secho("Only --retrieval-only is available until generation is built.", fg="red")
        raise typer.Exit(2)
    if mode is not None and mode not in MODES:
        raise typer.BadParameter(f"expected one of {', '.join(MODES)}", param_hint="--mode")

    modes = [mode] if mode else list(MODES)
    cases = read_cases(cases_path)
    by_mode = rev.run(cases, modes)
    for m, results in by_mode.items():
        r5, r8 = (rev.recall(results, k) for k in rev.KS)
        typer.echo(f"{m:<8} recall@5 {r5:6.1%}   recall@8 {r8:6.1%}   (n={len(results)})")
    typer.secho(f"Report: {rev.write_report(by_mode)}", fg="green")
