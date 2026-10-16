"""`nhl` command-line client. A thin wrapper over the HTTP API."""

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
