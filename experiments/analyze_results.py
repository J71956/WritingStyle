"""Tables, figures and significance tests for the paper (§4 of TEST-AND-PAPER-PLAN.md).

Reads artifacts/eval/campaign.jsonl (automated metric) and the ab_trial rows in
artifacts/eval_log.jsonl (human judgment), and writes:

    artifacts/eval/table_by_condition.csv     mean +/- std similarity per condition
    artifacts/eval/table_by_model.csv         ... per model (+ tokens/s)
    artifacts/eval/table_by_doc_type.csv      ... per doc_type x condition
    artifacts/eval/ablation_tests.csv         Wilcoxon signed-rank, paired by cell
    artifacts/eval/ab_summary.csv             binomial test + Wilson CI
    paper/figures/fig_ablation.(pdf|png)      condition means, grouped by model
    paper/figures/fig_by_doc_type.(pdf|png)   condition means, grouped by doc_type

Caveat carried into every table: the stylometric similarity is a *proxy*. It
measures distance to the author's feature vector, not whether a human finds the
text convincing. The A/B numbers are the ones that answer that question, and
they come from a single rater at n~20 — reported here with the interval, not as
a point estimate.

Usage:
    python experiments/analyze_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stylellm.config import load_settings

CONDITION_ORDER = ["plain", "profile_only", "exemplars_only", "full"]


# --- io ---------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


# --- stats ------------------------------------------------------------------


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well-behaved at the small n this study has."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --- analyses ---------------------------------------------------------------


def cell_key(row: dict) -> tuple:
    """Identity of a grid cell ignoring condition — the pairing key for Wilcoxon."""
    return (row["model_key"], row["prompt_id"], row["rep"])


def analyse_campaign(rows: list[dict], eval_dir: Path, fig_dir: Path) -> dict:
    usable = [r for r in rows if not r.get("empty")]
    print(f"Campaign: {len(rows)} rows, {len(usable)} non-empty "
          f"({len(rows) - len(usable)} empty excluded)")
    if not usable:
        print("  no usable rows — skipping campaign analysis")
        return {}

    by_condition: dict[str, list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    by_model_latency: dict[str, list[float]] = defaultdict(list)
    by_model_tps: dict[str, list[float]] = defaultdict(list)
    by_type_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model_cond: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in usable:
        sim = r["style_similarity"]
        by_condition[r["condition"]].append(sim)
        by_model[r["model_key"]].append(sim)
        by_model_latency[r["model_key"]].append(r["latency_ms"])
        if r.get("new_tokens") and r.get("latency_ms"):
            by_model_tps[r["model_key"]].append(1000 * r["new_tokens"] / r["latency_ms"])
        by_type_cond[(r["doc_type"], r["condition"])].append(sim)
        by_model_cond[(r["model_key"], r["condition"])].append(sim)

    conditions = [c for c in CONDITION_ORDER if c in by_condition]
    models = sorted(by_model)
    doc_types = sorted({dt for dt, _ in by_type_cond})

    write_csv(
        eval_dir / "table_by_condition.csv",
        ["condition", "n", "mean_similarity", "std"],
        [[c, len(by_condition[c]), *(f"{v:.4f}" for v in mean_std(by_condition[c]))]
         for c in conditions],
    )
    write_csv(
        eval_dir / "table_by_model.csv",
        ["model_key", "n", "mean_similarity", "std", "mean_latency_ms", "mean_tokens_per_s"],
        [[m, len(by_model[m]),
          *(f"{v:.4f}" for v in mean_std(by_model[m])),
          f"{mean_std(by_model_latency[m])[0]:.0f}",
          f"{mean_std(by_model_tps[m])[0]:.2f}" if by_model_tps[m] else "n/a"]
         for m in models],
    )
    write_csv(
        eval_dir / "table_by_doc_type.csv",
        ["doc_type", *conditions],
        [[dt, *(f"{mean_std(by_type_cond[(dt, c)])[0]:.4f}" if by_type_cond[(dt, c)] else "n/a"
                for c in conditions)]
         for dt in doc_types],
    )

    _ablation_tests(usable, conditions, eval_dir)
    _plot_grouped(fig_dir / "fig_ablation", by_model_cond, models, conditions,
                  "Model", "Mean stylometric similarity",
                  "Ablation ladder by model")
    _plot_grouped(fig_dir / "fig_by_doc_type", by_type_cond, doc_types, conditions,
                  "Document type", "Mean stylometric similarity",
                  "Ablation ladder by document type")

    return {"conditions": {c: mean_std(by_condition[c]) for c in conditions}}


def _ablation_tests(rows: list[dict], conditions: list[str], eval_dir: Path) -> None:
    """Paired Wilcoxon signed-rank of each condition against `full`.

    Pairing is by (model, prompt, rep): the same request under the same seed, so
    the only difference between the paired draws is what the prompt carried.
    Small, non-normal samples — hence the signed-rank test rather than a t-test.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  scipy not installed — skipping Wilcoxon (pip install -e '.[dev]')")
        return

    lookup: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        lookup[cell_key(r)][r["condition"]] = r["style_similarity"]

    out = []
    for cond in conditions:
        if cond == "full":
            continue
        pairs = [(v["full"], v[cond]) for v in lookup.values() if "full" in v and cond in v]
        pairs = [(a, b) for a, b in pairs if a != b]  # zero differences carry no rank
        if len(pairs) < 6:
            out.append([f"full vs {cond}", len(pairs), "n/a", "n/a", "insufficient pairs"])
            continue
        stat, p = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
        delta = mean_std([a - b for a, b in pairs])[0]
        out.append([f"full vs {cond}", len(pairs), f"{stat:.1f}", f"{p:.4g}",
                    f"mean delta {delta:+.4f}"])

    write_csv(eval_dir / "ablation_tests.csv",
              ["comparison", "n_pairs", "wilcoxon_W", "p_value", "note"], out)


def analyse_ab(rows: list[dict], eval_dir: Path) -> None:
    trials = [r for r in rows if r.get("type") == "ab_trial" and r.get("preferred")]
    print(f"A/B: {len(trials)} rated trials")
    if not trials:
        print("  none yet — run experiments/run_ab_session.py")
        return

    n = len(trials)
    k = sum(1 for t in trials if t["preferred"] == "styled")
    lo, hi = wilson_interval(k, n)
    try:
        from scipy.stats import binomtest

        p = binomtest(k, n, 0.5, alternative="two-sided").pvalue
        p_str = f"{p:.4g}"
    except ImportError:
        print("  scipy not installed — reporting the proportion without a p-value")
        p_str = "n/a"

    write_csv(
        eval_dir / "ab_summary.csv",
        ["n_trials", "styled_preferred", "proportion", "wilson_ci_low", "wilson_ci_high",
         "binomial_p_vs_0.5", "caveat"],
        [[n, k, f"{k / n:.4f}", f"{lo:.4f}", f"{hi:.4f}", p_str,
          "single self-rater; not blind to authorship of the study"]],
    )
    print(f"  styled preferred {k}/{n} ({100 * k / n:.0f}%), "
          f"95% CI [{lo:.2f}, {hi:.2f}], p={p_str}")


# --- figures ----------------------------------------------------------------


def _plot_grouped(stem: Path, data: dict, groups: list[str], conditions: list[str],
                  xlabel: str, ylabel: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures (pip install -e '.[dev]')")
        return

    width = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 4))
    for i, cond in enumerate(conditions):
        xs, ys, errs = [], [], []
        for j, g in enumerate(groups):
            vals = data.get((g, cond), [])
            if not vals:
                continue
            m, s = mean_std(vals)
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(m)
            errs.append(s)
        ax.bar(xs, ys, width, yerr=errs, capsize=2, label=cond)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
        print(f"  wrote {stem}.{ext}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--campaign", default=None, help="Path to campaign.jsonl")
    ap.add_argument("--figures", default=str(REPO_ROOT / "paper" / "figures"))
    args = ap.parse_args()

    settings = load_settings(args.config)
    eval_dir = settings.artifacts_path / "eval"
    campaign_path = Path(args.campaign) if args.campaign else eval_dir / "campaign.jsonl"

    analyse_campaign(read_jsonl(campaign_path), eval_dir, Path(args.figures))
    analyse_ab(read_jsonl(settings.artifacts_path / settings.evaluate.log_file), eval_dir)
    print("\nEvery number in the paper should trace back to a row under artifacts/eval/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
