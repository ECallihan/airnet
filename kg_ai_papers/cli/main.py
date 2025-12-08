# kg_ai_papers/cli/main.py

from __future__ import annotations

import typer
from kg_ai_papers.cli import query_cli, pipeline_cli

app = typer.Typer(help="CLI tools for the AI paper knowledge graph.")

app.add_typer(query_cli.app, name="query")
app.add_typer(pipeline_cli.app, name="pipeline")

if __name__ == "__main__":
    app()
