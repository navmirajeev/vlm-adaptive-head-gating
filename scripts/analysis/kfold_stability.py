"""K-fold stability of the attention head classification.

Head selection and the ablation evaluation both draw on the same set of
conflicting examples. This script checks whether the selected set depends on
which examples are used for discovery: the examples of each model are
partitioned into K folds, the PCA classification is re-run on the remaining
K-1 folds, and the selected head set is compared against the full-data
classification.

Per fold we record:
- exact recovery of the promoting set (set equality)
- Jaccard overlap against the full-data set, for promoting and suppressing
- the symmetric difference, so deviation *size* is visible and not assumed

Fold construction
-----------------
Examples are sorted by ``example_id``, shuffled with a fixed seed, then split
into K contiguous parts. ``--shuffler`` selects the RNG:

- ``python`` (default): ``random.Random(seed).shuffle``
- ``numpy``:  ``np.random.default_rng(seed).permutation``
- ``sklearn``: ``KFold(shuffle=True, random_state=seed)``

The choice changes which borderline heads move between folds, but not the
conclusion. Use ``--seed-sweep`` to see the spread across seeds.

Usage:
    PYTHONPATH=src:. python scripts/analysis/kfold_stability.py
    PYTHONPATH=src:. python scripts/analysis/kfold_stability.py --seed-sweep 20
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analysis._patching_common import (
    filter_results,
    load_correct_example_ids,
    load_patching_results,
    model_key,
    sort_models,
)
from scripts.analysis.classify_attn_heads import classify_attn_heads
from vdpo.settings import settings

PATCHING_DIR = settings.output_dir / "patching_last_token_attn_heads"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "kfold_stability.json"

SHUFFLERS = ("python", "numpy", "sklearn")


def jaccard(a: set, b: set) -> float:
    """Jaccard overlap; two empty sets count as perfect agreement."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def fold_indices(n: int, n_folds: int, seed: int, shuffler: str) -> list[set[int]]:
    """Partition 0..n-1 into n_folds held-out index sets."""
    if shuffler == "sklearn":
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [set(held.tolist()) for _, held in splitter.split(range(n))]

    if shuffler == "python":
        order = list(range(n))
        random.Random(seed).shuffle(order)
        order = np.array(order)
    elif shuffler == "numpy":
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"Unknown shuffler: {shuffler}")

    return [set(part.tolist()) for part in np.array_split(order, n_folds)]


def kfold_stability(
    results: list[dict],
    n_folds: int = 5,
    seed: int = 42,
    sigma_threshold: float = 2.0,
    shuffler: str = "python",
) -> dict[str, dict]:
    """Per model, re-classify on each K-1 fold subset and compare to full data."""
    grouped = defaultdict(list)
    for r in results:
        grouped[model_key(r)].append(r)

    report = {}
    for model in sort_models(list(grouped.keys())):
        # Stable example order before shuffling, so the seed fully determines folds.
        rs = sorted(grouped[model], key=lambda r: r["example_id"])

        full = classify_attn_heads(rs, sigma_threshold=sigma_threshold)[model]
        full_promoting = {tuple(h) for h in full["promoting"]}
        full_suppressing = {tuple(h) for h in full["suppressing"]}

        folds = []
        for held_out in fold_indices(len(rs), n_folds, seed, shuffler):
            subset = [r for i, r in enumerate(rs) if i not in held_out]

            sub = classify_attn_heads(subset, sigma_threshold=sigma_threshold)[model]
            promoting = {tuple(h) for h in sub["promoting"]}
            suppressing = {tuple(h) for h in sub["suppressing"]}

            folds.append(
                {
                    "n_discovery": len(subset),
                    "n_held_out": len(held_out),
                    "n_promoting": len(promoting),
                    "n_suppressing": len(suppressing),
                    "exact_promoting": promoting == full_promoting,
                    "exact_suppressing": suppressing == full_suppressing,
                    "jaccard_promoting": jaccard(promoting, full_promoting),
                    "jaccard_suppressing": jaccard(suppressing, full_suppressing),
                    "n_promoting_differing": len(promoting ^ full_promoting),
                    "promoting_only_in_fold": sorted(promoting - full_promoting),
                    "promoting_only_in_full": sorted(full_promoting - promoting),
                }
            )

        report[model] = {
            "n_examples": len(rs),
            "n_promoting_full": len(full_promoting),
            "n_suppressing_full": len(full_suppressing),
            "folds_exact_promoting": sum(f["exact_promoting"] for f in folds),
            "n_folds": n_folds,
            "max_promoting_differing": max(f["n_promoting_differing"] for f in folds),
            "jaccard_promoting_min": min(f["jaccard_promoting"] for f in folds),
            "jaccard_promoting_max": max(f["jaccard_promoting"] for f in folds),
            "jaccard_suppressing_min": min(f["jaccard_suppressing"] for f in folds),
            "jaccard_suppressing_max": max(f["jaccard_suppressing"] for f in folds),
            "folds": folds,
        }

    return report


def _range_str(lo: float, hi: float) -> str:
    return f"{lo:.2f}" if lo == hi else f"{lo:.2f} to {hi:.2f}"


def print_report(report: dict[str, dict]) -> None:
    print(f"\n{'=' * 82}")
    print("  K-FOLD CLASSIFICATION STABILITY (re-classify on K-1 folds)")
    print(f"{'=' * 82}\n")

    print(
        f"  | {'Model':<16} | {'Folds exact':<11} | {'Jaccard (prom.)':<15}"
        f" | {'Jaccard (supp.)':<15} |"
    )
    print(f"  |{'-' * 18}|{'-' * 13}|{'-' * 17}|{'-' * 17}|")

    total_exact = 0
    for model in sort_models(list(report.keys())):
        r = report[model]
        total_exact += r["folds_exact_promoting"]
        exact = f"{r['folds_exact_promoting']}/{r['n_folds']}"
        print(
            f"  | {model:<16} | {exact:<11} | "
            f"{_range_str(r['jaccard_promoting_min'], r['jaccard_promoting_max']):<15} | "
            f"{_range_str(r['jaccard_suppressing_min'], r['jaccard_suppressing_max']):<15} |"
        )

    n_runs = sum(r["n_folds"] for r in report.values())
    print(
        f"\n  Promoting set recovered exactly in {total_exact} of {n_runs}"
        " model-fold runs."
    )

    worst = max(r["max_promoting_differing"] for r in report.values())
    print(f"  Largest promoting-set deviation in any single fold: {worst} head(s).")

    print("\n  Deviations (promoting heads differing from the full-data set):")
    any_dev = False
    for model in sort_models(list(report.keys())):
        for i, f in enumerate(report[model]["folds"]):
            if f["exact_promoting"]:
                continue
            any_dev = True
            extra = [f"L{layer}H{head}" for layer, head in f["promoting_only_in_fold"]]
            missing = [
                f"L{layer}H{head}" for layer, head in f["promoting_only_in_full"]
            ]
            print(
                f"    {model:<16} fold {i}: {f['n_promoting_differing']} head(s)"
                f"  +{extra or '[]'}  -{missing or '[]'}"
            )
    if not any_dev:
        print("    none")


def seed_sweep(
    results: list[dict],
    n_seeds: int,
    n_folds: int,
    sigma_threshold: float,
    shuffler: str,
) -> None:
    """Report how much the headline number moves across seeds."""
    totals, worst, per_model = [], 0, defaultdict(list)
    for seed in range(n_seeds):
        report = kfold_stability(results, n_folds, seed, sigma_threshold, shuffler)
        totals.append(sum(r["folds_exact_promoting"] for r in report.values()))
        worst = max(worst, max(r["max_promoting_differing"] for r in report.values()))
        for m, r in report.items():
            per_model[m].append(r["folds_exact_promoting"])

    n_runs = n_folds * len(per_model)
    print(f"\n{'=' * 82}")
    print(f"  SEED SENSITIVITY ({n_seeds} seeds, shuffler={shuffler})")
    print(f"{'=' * 82}\n")
    print(
        f"  Exact recovery: min={min(totals)}/{n_runs} max={max(totals)}/{n_runs}"
        f" mean={sum(totals) / len(totals):.1f}/{n_runs}"
    )
    print(f"  Largest deviation in any model-fold run: {worst} head(s)\n")
    for m in sort_models(list(per_model.keys())):
        v = per_model[m]
        print(
            f"    {m:<16} folds exact: min={min(v)}/{n_folds} max={max(v)}/{n_folds}"
            f" mean={sum(v) / len(v):.1f}/{n_folds}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--shuffler", choices=SHUFFLERS, default="python")
    parser.add_argument(
        "--seed-sweep",
        type=int,
        default=0,
        metavar="N",
        help="Also report the spread over N seeds (0 disables).",
    )
    args = parser.parse_args()

    print(f"Loading attention head patching results from {PATCHING_DIR} ...")
    all_results = load_patching_results(PATCHING_DIR)
    results = filter_results(all_results, load_correct_example_ids())
    print(f"Filtered to {len(results)} examples.")

    report = kfold_stability(
        results,
        n_folds=args.n_folds,
        seed=args.seed,
        sigma_threshold=args.sigma,
        shuffler=args.shuffler,
    )
    print_report(report)

    if args.seed_sweep:
        seed_sweep(results, args.seed_sweep, args.n_folds, args.sigma, args.shuffler)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_folds": args.n_folds,
                "sigma": args.sigma,
                "shuffler": args.shuffler,
                "models": report,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved k-fold report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
