"""Grid runner for the style-generation test campaign (§2 of TEST-AND-PAPER-PLAN.md).

Sweeps  model x prompt x ablation-condition x repetition  and appends one JSON
row per draft to artifacts/eval/campaign.jsonl.

Everything scientific is reused from the library — retrieval, prompt assembly,
the backend, and the stylometric metric. This script only orchestrates and logs.

Two properties matter for the paper's reproducibility claim:

* **Model-outermost iteration.** Each set of HF weights is loaded exactly once
  and reused across its whole slice of the grid; reloading an 8B model per cell
  would dominate the runtime.
* **Per-rep seeding.** `set_seed(seed + rep)` runs before every generation, so
  temperature still samples but each (cell, rep) is individually reproducible —
  something the Ollama path could not offer.

Usage:
    # offline dry run — proves the grid, conditions, scoring and logging
    python experiments/run_campaign.py --backend fake --limit 8

    # the real campaign
    HF_HUB_OFFLINE=1 python experiments/run_campaign.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stylellm import evaluate
from stylellm import generate as gen
from stylellm.config import load_settings, set_seeds
from stylellm.index import load_index
from stylellm.models import Document, StyleProfile
from stylellm.retrieve import retrieve

EXPERIMENTS_DIR = Path(__file__).resolve().parent

# The ablation ladder: which of the two style components each condition carries.
CONDITIONS: dict[str, dict[str, bool]] = {
    "plain": {"use_profile": False, "use_exemplars": False},
    "profile_only": {"use_profile": True, "use_exemplars": False},
    "exemplars_only": {"use_profile": False, "use_exemplars": True},
    "full": {"use_profile": True, "use_exemplars": True},
}


# --- corpus / profile loading (mirrors cli._load_corpus / _load_profile) -----


def load_corpus(settings) -> list[Document]:
    corpus_dir = settings.artifacts_path / "corpus"
    jsonl = corpus_dir / "documents.jsonl"
    if not jsonl.exists():
        raise SystemExit("No ingested corpus found. Run `stylellm ingest` first.")
    documents: list[Document] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        text = (corpus_dir / f"{meta['doc_id']}.txt").read_text(encoding="utf-8")
        documents.append(Document(text=text, **meta))
    return documents


def load_profile(settings) -> StyleProfile:
    path = settings.artifacts_path / "style_profile.json"
    if not path.exists():
        raise SystemExit("No style profile found. Run `stylellm analyze` first.")
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


# --- grid -------------------------------------------------------------------


def load_prompts(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["prompts"]


def load_models(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["models"]


def run_id(model_key: str, prompt_id: str, condition: str, rep: int) -> str:
    """Stable identity for one grid cell — the basis of resumability."""
    return f"{model_key}|{prompt_id}|{condition}|{rep}"


def completed_run_ids(out_path: Path) -> set[str]:
    """Read back run_ids already logged, so a re-run resumes instead of duplicating."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["run_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # tolerate a partially-written trailing row
    return done


def make_backend_for(model_spec: dict, settings, backend_override: str | None):
    """Build the backend for one model slice, honouring an offline override.

    For the llamacpp default the harness owns the server process: one
    llama-server per model slice, started here and stopped by `_release`. It is
    attached to the backend as `_server` so the slice's cleanup path is the same
    shape as unloading HF weights.
    """
    if backend_override in ("fake", "ollama"):
        settings.generate.backend = backend_override
        if backend_override == "ollama":
            settings.generate.model = model_spec["model"]
        return gen.make_backend(settings)

    if backend_override == "hf":
        return gen.HFBackend(
            model_spec["model"],
            device=settings.generate.device,
            dtype=settings.generate.dtype,
            enable_thinking=model_spec.get("enable_thinking", False),
            load_in_4bit=model_spec.get("load_in_4bit", False),
            revision=model_spec.get("revision"),
        )

    from stylellm.serve import LlamaServer

    cfg = settings.generate
    server = LlamaServer(
        model_spec["hf_spec"],
        port=int(cfg.llamacpp_host.rsplit(":", 1)[-1]),
        n_gpu_layers=cfg.n_gpu_layers,
        ctx_size=cfg.ctx_size,
        binary=cfg.llamacpp_bin,
    ).start()
    try:
        backend = gen.LlamaCppBackend(
            server.host,
            gguf_file=model_spec.get("gguf_file"),
            enable_thinking=model_spec.get("enable_thinking", False),
        )
    except Exception:
        server.stop()  # don't strand the process if the identity check fails
        raise
    backend._server = server
    return backend


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="Path to config YAML")
    ap.add_argument("--prompts", default=str(EXPERIMENTS_DIR / "prompts.yaml"))
    ap.add_argument("--models", default=str(EXPERIMENTS_DIR / "models.yaml"))
    ap.add_argument("--backend", default=None, choices=["fake", "ollama", "hf", "llamacpp"],
                    help="Override the backend (use 'fake' for an offline dry run); "
                         "default llamacpp starts a llama-server per model slice")
    ap.add_argument("--model-key", action="append", default=None,
                    help="Restrict to these model keys (repeatable)")
    ap.add_argument("--reps", type=int, default=3, help="Repetitions per cell")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N generations")
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default artifacts/eval/campaign.jsonl)")
    args = ap.parse_args()

    settings = load_settings(args.config)
    set_seeds(settings.seed)

    out_path = Path(args.out) if args.out else settings.artifacts_path / "eval" / "campaign.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    models = load_models(Path(args.models))
    if args.model_key:
        models = [m for m in models if m["key"] in set(args.model_key)]
        if not models:
            raise SystemExit(f"No models matched {args.model_key}")

    documents = load_corpus(settings)
    profile = load_profile(settings)
    index = load_index(settings)

    # Computed once from the corpus: the standardized-Euclidean form of the
    # metric, not the self-contained fallback the CLI falls back to (R6.1).
    scales = evaluate.feature_scales(documents, settings)

    done = completed_run_ids(out_path)
    total = len(models) * len(prompts) * len(CONDITIONS) * args.reps
    print(f"Grid: {len(models)} models x {len(prompts)} prompts x "
          f"{len(CONDITIONS)} conditions x {args.reps} reps = {total} generations")
    print(f"Already logged: {len(done)} — resuming into {out_path}")

    n_run = 0
    for spec in models:  # model-outermost: load each set of weights once
        pending = [
            (p, c, r)
            for p in prompts
            for c in CONDITIONS
            for r in range(args.reps)
            if run_id(spec["key"], p["id"], c, r) not in done
        ]
        if not pending:
            print(f"[{spec['key']}] nothing pending, skipping model load.")
            continue

        print(f"\n[{spec['key']}] loading {spec['model']} ({len(pending)} cells pending)...")
        t0 = time.perf_counter()
        try:
            backend = make_backend_for(spec, settings, args.backend)
        except Exception as e:
            print(f"[{spec['key']}] SKIPPED — could not load backend: {e}")
            continue
        revision = getattr(backend, "revision", spec.get("revision")) or "unknown"
        print(f"[{spec['key']}] ready in {time.perf_counter() - t0:.1f}s (revision={revision})")

        # Retrieval depends only on (prompt, doc_type), so cache per prompt id.
        exemplar_cache: dict[str, list] = {}
        empty_streak = 0

        for prompt_spec, condition, rep in pending:
            pid = prompt_spec["id"]
            doc_type = prompt_spec["doc_type"]
            text_req = prompt_spec["prompt"]
            rid = run_id(spec["key"], pid, condition, rep)

            if pid not in exemplar_cache:
                exemplar_cache[pid] = retrieve(text_req, index, settings, doc_type=doc_type)
            exemplars = exemplar_cache[pid]

            seed = settings.seed + rep
            _set_generation_seed(seed, args.backend, backend)
            prompt_text = gen.assemble_prompt(
                text_req, profile, exemplars, doc_type, **CONDITIONS[condition]
            )

            t = time.perf_counter()
            try:
                draft = backend.complete(
                    prompt_text, settings.generate.max_tokens, settings.generate.temperature
                )
            except Exception as e:
                print(f"  {rid}: ERROR {e}")
                draft = ""
            latency_ms = int((time.perf_counter() - t) * 1000)

            empty = not draft.strip()
            similarity = (
                0.0 if empty
                else evaluate.stylometric_similarity(
                    draft, profile, doc_type, settings, scales=scales
                )
            )

            row = {
                "run_id": rid,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_key": spec["key"],
                "model": spec["model"],
                "model_revision": revision,
                "doc_type": doc_type,
                "prompt_id": pid,
                "condition": condition,
                "rep": rep,
                "seed": seed,
                "style_similarity": similarity,
                "latency_ms": latency_ms,
                "prompt_tokens": getattr(backend, "last_prompt_tokens", None),
                "new_tokens": getattr(backend, "last_new_tokens", None),
                "empty": empty,
                # Exemplars are recorded for every condition; whether they were
                # actually injected is determined by `condition`.
                "exemplars_used": [e.chunk.chunk_id for e in exemplars],
                "exemplars_injected": CONDITIONS[condition]["use_exemplars"] and bool(exemplars),
                "text": draft,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

            n_run += 1
            flag = " [EMPTY]" if empty else ""
            print(f"  {rid}: sim={similarity:.3f} {latency_ms}ms{flag}")

            # Guard from the plan: a model that reliably returns nothing gets
            # flagged and skipped rather than silently logging 0.0 similarities.
            empty_streak = empty_streak + 1 if empty else 0
            if empty_streak >= 8:
                print(f"[{spec['key']}] ABORTED — 8 consecutive empty outputs. "
                      f"Check the chat template / enable_thinking setting.")
                break

            if args.limit and n_run >= args.limit:
                print(f"\nStopping at --limit {args.limit}.")
                _release(backend)  # never strand a llama-server on the way out
                return 0

        _release(backend)

    print(f"\nDone: {n_run} new generations -> {out_path}")
    return 0


def _set_generation_seed(seed: int, backend_override: str | None, backend=None) -> None:
    """Pin the sampler before each generation (the reproducibility claim, P10)."""
    if backend_override == "fake":
        return  # deterministic already; avoids importing transformers offline
    if backend is not None and hasattr(backend, "seed"):
        # llama.cpp samples in the server process, out of reach of set_seed —
        # the seed rides along with each request instead.
        backend.seed = seed
        return
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        set_seeds(seed)


def _release(backend) -> None:
    """Free GPU memory before loading the next model."""
    server = getattr(backend, "_server", None)
    if server is not None:
        server.stop()  # the weights live in that process, not this one
        return
    if not hasattr(backend, "model"):
        return
    try:
        import torch

        del backend.model
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
