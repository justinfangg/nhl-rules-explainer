"""`nhl` command-line client. A thin wrapper over the HTTP API."""

import re
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer

from nhl_rules.config import get_settings

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
