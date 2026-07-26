"""Local single-user CLI (R5.3-style entrypoint for the v1 analysis half).

Commands:
  stylellm ingest    -> artifacts/corpus/ + artifacts/ingestion_report.md
  stylellm analyze   -> artifacts/style_profile.json + artifacts/style_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .config import load_settings, set_seeds
from .ingest import ingest_corpus, persist_corpus
from .models import Document
from .report import render_ingestion_report
from .style_analyzer import analyze, persist_profile

app = typer.Typer(help="Personal Style LLM — local writing-style analysis (v1).")
console = Console()


@app.command()
def ingest(config: str = typer.Option(None, help="Path to config YAML (default: config/default.yaml)")):
    """Extract, clean, tag, and flag the corpus; write the ingestion report."""
    settings = load_settings(config)
    set_seeds(settings.seed)
    result = ingest_corpus(settings)
    corpus_dir = persist_corpus(result, settings)
    report_md = render_ingestion_report(result)
    report_path = settings.artifacts_path / "ingestion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    total = sum(d.token_count for d in result.documents)
    dominant = [d.doc_id for d in result.documents if d.is_dominant]
    console.print(f"[green]Ingested[/] {len(result.documents)} docs, {total:,} tokens.")
    console.print(f"Dominant: {dominant or 'none'}")
    if result.warnings:
        console.print(f"[yellow]{len(result.warnings)} extraction warning(s)[/] — see report.")
    console.print(f"Cleaned corpus: {corpus_dir}")
    console.print(f"Report: {report_path}")


@app.command("analyze")
def analyze_cmd(config: str = typer.Option(None, "--config", help="Path to config YAML")):
    """Compute the skew-aware StyleProfile and render the style report."""
    settings = load_settings(config)
    set_seeds(settings.seed)

    jsonl = settings.artifacts_path / "corpus" / "documents.jsonl"
    corpus_dir = settings.artifacts_path / "corpus"
    if not jsonl.exists():
        console.print("[red]No ingested corpus found.[/] Run `stylellm ingest` first.")
        raise typer.Exit(code=1)

    documents: list[Document] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        text = (corpus_dir / f"{meta['doc_id']}.txt").read_text(encoding="utf-8")
        documents.append(Document(text=text, **meta))

    min_tok = settings.analyzer.min_doc_tokens
    excluded = [d for d in documents if d.token_count < min_tok]
    documents = [d for d in documents if d.token_count >= min_tok]
    if excluded:
        names = ", ".join(f"{d.doc_id} ({d.token_count} tok)" for d in excluded)
        console.print(f"[yellow]Excluding {len(excluded)} doc(s) below {min_tok} tokens:[/] {names}")

    console.print(f"Analyzing {len(documents)} documents...")
    profile = analyze(documents, settings)
    json_path, md_path = persist_profile(profile, settings)
    console.print(f"[green]StyleProfile v{profile.version}[/] written.")
    console.print(f"Profile: {json_path}")
    console.print(f"Report:  {md_path}")


if __name__ == "__main__":
    app()
