"""Batch blind-A/B session — the real acceptance gate (§3 of TEST-AND-PAPER-PLAN.md, R6.2).

A thin loop over the *existing* single-trial path: `evaluate.run_ab_trial` drafts
styled-vs-plain with the labels already randomized, and `evaluate.record_preference`
resolves the author's keypress and appends an `ab_trial` row to
artifacts/eval_log.jsonl — exactly what `stylellm ab` does. No new evaluation
logic is introduced here, only the sampling and the loop.

The automated metric in run_campaign.py is a proxy; this is the human judgment
the paper's headline number comes from.

Usage:
    python experiments/run_ab_session.py --trials 20
    python experiments/run_ab_session.py --trials 4 --backend fake --auto A   # smoke test
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_campaign import EXPERIMENTS_DIR, load_profile, load_prompts

from stylellm.config import load_settings, set_seeds
from stylellm.evaluate import append_log, record_preference, run_ab_trial
from stylellm.index import load_index


def sample_balanced(prompts: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n prompts spread as evenly as possible across doc_types."""
    by_type: dict[str, list[dict]] = {}
    for p in prompts:
        by_type.setdefault(p["doc_type"], []).append(p)
    for group in by_type.values():
        rng.shuffle(group)

    picked: list[dict] = []
    # Round-robin over the types so no genre dominates a 20-trial session.
    while len(picked) < n:
        added = False
        for doc_type in sorted(by_type):
            if len(picked) >= n:
                break
            group = by_type[doc_type]
            if group:
                picked.append(group.pop())
                added = True
        if not added:  # exhausted the bank — cycle it again
            for p in prompts:
                by_type.setdefault(p["doc_type"], []).append(p)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=str(EXPERIMENTS_DIR / "prompts.yaml"))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--backend", default=None, help="Override backend (hf | ollama | fake)")
    ap.add_argument("--seed", type=int, default=None, help="Override the sampling seed")
    ap.add_argument("--auto", default=None, choices=["A", "B"],
                    help="Non-interactive: always pick this label (smoke tests only — "
                         "produces meaningless preference data)")
    args = ap.parse_args()

    settings = load_settings(args.config)
    seed = args.seed if args.seed is not None else settings.seed
    set_seeds(seed)
    if args.backend:
        settings.generate.backend = args.backend

    profile = load_profile(settings)
    index = load_index(settings)
    prompts = sample_balanced(load_prompts(Path(args.prompts)), args.trials, random.Random(seed))

    # Record the session's parameters alongside the trials, so the paper can say
    # which model and seed produced the human numbers.
    append_log(
        {"type": "ab_session_start", "model": settings.generate.model,
         "backend": settings.generate.backend, "seed": seed, "trials": args.trials},
        settings,
    )

    print(f"Blind A/B — {args.trials} trials, model={settings.generate.model}, seed={seed}")
    print("For each pair, pick whichever reads more like your own writing.\n")

    counts = {"styled": 0, "plain": 0}
    for i, spec in enumerate(prompts, 1):
        trial, presentation = run_ab_trial(
            spec["prompt"], profile, index, settings, doc_type=spec["doc_type"]
        )
        print(f"\n=== Trial {i}/{args.trials} — {spec['doc_type']} ({spec['id']}) ===")
        print(f"Request: {spec['prompt'].strip()}\n")
        print(f"--- A ---\n{getattr(trial, 'output_' + presentation['A'])}\n")
        print(f"--- B ---\n{getattr(trial, 'output_' + presentation['B'])}\n")

        choice = args.auto
        while choice not in ("A", "B"):
            choice = input("Which reads more like you? (A/B): ").strip().upper()

        trial = record_preference(trial, presentation, choice, settings)
        counts[trial.preferred] += 1
        print(f"  recorded: {trial.preferred}")

    n = counts["styled"] + counts["plain"]
    pct = 100 * counts["styled"] / n if n else 0.0
    print(f"\nSession complete: styled preferred {counts['styled']}/{n} ({pct:.0f}%).")
    print(f"Logged to {settings.artifacts_path / settings.evaluate.log_file}")
    print("Run experiments/analyze_results.py for the binomial test and CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
