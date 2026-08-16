"""Local single-user CLI (R5.3-style entrypoint for the v1 analysis half).

Commands:
  stylellm ingest    -> artifacts/corpus/ + artifacts/ingestion_report.md
  stylellm analyze   -> artifacts/style_profile.json + artifacts/style_report.md
  stylellm index     -> artifacts/index/ (chunks.jsonl + embeddings.npy)
  stylellm generate  -> draft text in the author's voice (stdout)
  stylellm ab        -> blind A/B trial (styled vs. plain), logged to eval_log.jsonl
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from .config import Settings, load_settings, set_seeds
from .ingest import ingest_corpus, persist_corpus
from .models import Document, StyleProfile
from .report import render_ingestion_report
from .style_analyzer import analyze, persist_profile

app = typer.Typer(help="Personal Style LLM — local writing-style analysis + generation (v1).")
console = Console()


def _load_corpus(settings: Settings) -> list[Document]:
    """Load the ingested corpus from artifacts/corpus/ (shared by several commands)."""
    corpus_dir = settings.artifacts_path / "corpus"
    jsonl = corpus_dir / "documents.jsonl"
    if not jsonl.exists():
        console.print("[red]No ingested corpus found.[/] Run `stylellm ingest` first.")
        raise typer.Exit(code=1)
    documents: list[Document] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        text = (corpus_dir / f"{meta['doc_id']}.txt").read_text(encoding="utf-8")
        documents.append(Document(text=text, **meta))
    return documents


def _load_profile(settings: Settings) -> StyleProfile:
    """Load the persisted StyleProfile (shared by generate/ab)."""
    path = settings.artifacts_path / "style_profile.json"
    if not path.exists():
        console.print("[red]No style profile found.[/] Run `stylellm analyze` first.")
        raise typer.Exit(code=1)
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


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

    documents = _load_corpus(settings)

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


@app.command("index")
def index_cmd(config: str = typer.Option(None, "--config", help="Path to config YAML")):
    """Chunk the corpus, embed, and build the local in-memory index (R3)."""
    from .index import build_index, persist_index

    settings = load_settings(config)
    set_seeds(settings.seed)
    documents = _load_corpus(settings)

    console.print(f"Chunking + embedding {len(documents)} documents...")
    index = build_index(documents, settings)
    if index.embeddings is None:
        console.print("[yellow]Embeddings unavailable[/] (model not loadable) — "
                      "chunks indexed without vectors; retrieval will be disabled.")
    out_dir = persist_index(index, settings)
    console.print(f"[green]Indexed {len(index)} chunks.[/]")
    console.print(f"Index: {out_dir}")


@app.command("generate")
def generate_cmd(
    prompt: str = typer.Argument(..., help="What to write (e.g. 'a cover letter for a data analyst role')"),
    doc_type: str = typer.Option(None, "--doc-type", help="Filter exemplars to a doc_type"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override generation length"),
    backend: str = typer.Option(None, "--backend", help="Override backend: 'hf', 'ollama', or 'fake'"),
    config: str = typer.Option(None, "--config", help="Path to config YAML"),
):
    """Draft text in the author's voice: profile summary + retrieved exemplars (R5)."""
    from .generate import generate
    from .index import load_index
    from .evaluate import log_generation
    from .retrieve import retrieve

    settings = load_settings(config)
    set_seeds(settings.seed)
    if max_tokens:
        settings.generate.max_tokens = max_tokens
    if backend:
        settings.generate.backend = backend

    profile = _load_profile(settings)
    index = load_index(settings)
    exemplars = retrieve(prompt, index, settings, doc_type=doc_type)
    if not exemplars:
        console.print("[yellow]No exemplars retrieved[/] — generating from the style profile alone (R5.4).")

    result = generate(prompt, profile, exemplars, settings, doc_type=doc_type)
    log_generation(result, prompt, doc_type, settings)

    console.print(f"\n[bold]Draft[/] (style_similarity={result.style_similarity:.3f}, "
                  f"{len(result.exemplars_used)} exemplar(s), {result.latency_ms} ms):\n")
    console.print(result.generated_text)


@app.command("ab")
def ab_cmd(
    prompt: str = typer.Argument(..., help="Prompt to run styled-vs-plain"),
    doc_type: str = typer.Option(None, "--doc-type", help="Filter exemplars to a doc_type"),
    prefer: str = typer.Option(None, "--prefer", help="Record preference non-interactively: 'A' or 'B'"),
    backend: str = typer.Option(None, "--backend", help="Override backend: 'hf', 'ollama', or 'fake'"),
    config: str = typer.Option(None, "--config", help="Path to config YAML"),
):
    """Blind A/B trial: styled (profile + exemplars) vs. plain, logged for review (R6.2)."""
    from .index import load_index
    from .evaluate import record_preference, run_ab_trial

    settings = load_settings(config)
    set_seeds(settings.seed)
    if backend:
        settings.generate.backend = backend
    profile = _load_profile(settings)
    index = load_index(settings)

    trial, presentation = run_ab_trial(prompt, profile, index, settings, doc_type=doc_type)
    console.print("[bold]Blind A/B — labels hidden.[/]\n")
    console.print(f"[bold]A:[/]\n{getattr(trial, 'output_' + presentation['A'])}\n")
    console.print(f"[bold]B:[/]\n{getattr(trial, 'output_' + presentation['B'])}\n")

    choice = prefer or typer.prompt("Which reads more like the author? (A/B)")
    trial = record_preference(trial, presentation, choice, settings)
    console.print(f"[green]Recorded[/] — you preferred the [bold]{trial.preferred}[/] arm.")
    console.print(f"Logged to: {settings.artifacts_path / settings.evaluate.log_file}")


if __name__ == "__main__":
    app()
