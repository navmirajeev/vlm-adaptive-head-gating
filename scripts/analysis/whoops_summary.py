"""Five-model WHOOPS summary for the camera-ready generalization section.

Aggregates the WHOOPS pipeline outputs into the two tables section 4.7 needs:

1. **Circuit recovery.** How many of the attention heads classified on
   Visual-Counterfact are independently re-discovered when the classification is
   re-run from scratch on WHOOPS. This is the answer to the objection that the
   identified heads merely write color answer tokens: WHOOPS answers are
   open-vocabulary and non-color, so a pure color writer could not mediate them.

2. **Knockout ablation asymmetry.** Prior- and visual-grounding flip rates for
   the group ablations, with a McNemar exact test on the paired per-example
   outcomes.

Unlike the color task there is no correctness filter. The WHOOPS subset is
already flip-filtered upstream (only examples whose top-1 differs between the
two prompts enter the pipeline), so every example with data is used. This
matches ``classify_whoops.py``.

Note that WHOOPS outputs live under ``outputs/whoops/<runner>/`` rather than the
``outputs/<runner>/<contrast>/`` layout the ``--contrast`` flag of the other
knockout scripts expects, so this script resolves its own paths.

Usage:
    PYTHONPATH=src:. python scripts/analysis/whoops_summary.py
    PYTHONPATH=src:. python scripts/analysis/whoops_summary.py --latex
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from scripts.analysis._patching_common import sort_models
from scripts.analysis.kfold_stability import jaccard
from vdpo.settings import settings

WHOOPS_DIR = settings.output_dir / "whoops"
KNOCKOUT_DIR = WHOOPS_DIR / "knockout_attn_heads"
PATCHING_DIR = WHOOPS_DIR / "patching_last_token_attn_heads"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

COLOR_CLASSIFICATIONS = DATA_DIR / "classify_attn_heads.json"
WHOOPS_CLASSIFICATIONS = DATA_DIR / "classify_attn_heads_whoops.json"

# Side analysis: the color-selected circuit ablated on WHOOPS examples. Not part
# of the reported results, and it nests one level deeper than the model
# directories, so it must not be read as a model.
EXCLUDED_DIRS = {"frozen_color"}

GROUP_KNOCKOUTS = ("grp_promoting", "grp_suppressing")


def head_set(entries: list[dict]) -> set[tuple[int, int]]:
    """Normalize serialized classification entries to (layer, head) pairs."""
    return {(e["layer"], e["head"]) for e in entries}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pair counts.

    Under the null the two flip directions are equally likely, so the number of
    discordant pairs falling one way is Binomial(b + c, 1/2). Returns 1.0 when
    there are no discordant pairs, and is clamped to 1.0 when doubling the tail
    would otherwise exceed it.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compute_recovery(color: dict, whoops: dict) -> dict[str, dict]:
    """Per model, how much of the color circuit the WHOOPS classification finds.

    Models absent from either classification are skipped rather than reported as
    zero recovery, so a partially populated run does not look like a failure.
    """
    report = {}
    for model in sort_models(list(color.keys())):
        if model not in whoops:
            continue
        cp, wp = (
            head_set(color[model]["promoting"]),
            head_set(whoops[model]["promoting"]),
        )
        cs, ws = (
            head_set(color[model]["suppressing"]),
            head_set(whoops[model]["suppressing"]),
        )
        report[model] = {
            "n_promoting_color": len(cp),
            "n_promoting_whoops": len(wp),
            "promoting_recovered": len(cp & wp),
            "jaccard_promoting": jaccard(cp, wp),
            "n_suppressing_color": len(cs),
            "n_suppressing_whoops": len(ws),
            "suppressing_recovered": len(cs & ws),
            "jaccard_suppressing": jaccard(cs, ws),
        }
    return report


def load_whoops_knockout_results(knockout_dir: Path) -> list[dict]:
    """Load WHOOPS attention head knockout ablation results into a flat list."""
    results = []
    for json_path in sorted(knockout_dir.rglob("*.json")):
        parts = json_path.relative_to(knockout_dir).parts
        if parts[0] in EXCLUDED_DIRS:
            continue
        with json_path.open() as f:
            data = json.load(f)
        data["model_family"], data["model_size"] = parts[0], parts[1]
        results.append(data)
    return results


def load_patching_example_ids(patching_dir: Path) -> dict[str, set[str]]:
    """Per model, the example ids that have head-patching data.

    These are exactly the examples the classification saw, so they define which
    examples are eligible for the ablation summary.
    """
    ids = defaultdict(set)
    for json_path in sorted(patching_dir.rglob("*.json")):
        parts = json_path.relative_to(patching_dir).parts
        if parts[0] in EXCLUDED_DIRS:
            continue
        ids[f"{parts[0]}/{parts[1]}"].add(json_path.stem)
    return dict(ids)


def restrict_to_classified(
    results: list[dict], allowed: dict[str, set[str]]
) -> list[dict]:
    """Drop ablation examples that never entered the classification.

    A handful of examples have ablation outputs but no patching outputs, so they
    could not contribute to head selection. Evaluating the circuit on them would
    mix two different example sets within one model.
    """
    return [
        r
        for r in results
        if r["example_id"]
        in allowed.get(f"{r['model_family']}/{r['model_size']}", set())
    ]


def compute_ablation_summary(results: list[dict]) -> dict[str, dict]:
    """Per model and knockout config, flip rates plus the paired McNemar test.

    A prior-grounding flip is an example whose post-ablation prediction under the
    prior prompt is no longer the prior answer, and conversely for visual. The
    McNemar test pairs the two outcomes within each example, so it asks whether
    ablation moves prior grounding more than visual grounding on the same items.
    """
    by_model = defaultdict(lambda: defaultdict(list))
    for r in results:
        model = f"{r['model_family']}/{r['model_size']}"
        by_model[model][r["knockout_id"]].append(r)

    summary = {}
    for model in sort_models(list(by_model.keys())):
        summary[model] = {}
        for knockout_id, rs in by_model[model].items():
            n = len(rs)
            prior_flips = visual_flips = prior_only = visual_only = 0
            for r in rs:
                pf = r["prior_grounding"]["predicted"] != "prior"
                vf = r["visual_grounding"]["predicted"] != "visual"
                prior_flips += pf
                visual_flips += vf
                prior_only += pf and not vf
                visual_only += vf and not pf
            summary[model][knockout_id] = {
                "n": n,
                "prior_flips": prior_flips,
                "visual_flips": visual_flips,
                "prior_flip_rate": prior_flips / n if n else 0.0,
                "visual_flip_rate": visual_flips / n if n else 0.0,
                "prior_only": prior_only,
                "visual_only": visual_only,
                "mcnemar_p": mcnemar_exact(prior_only, visual_only),
            }
    return summary


def _p_str(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def print_recovery(recovery: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print("  CIRCUIT RECOVERY (classification re-run from scratch on WHOOPS)")
    print(f"{'=' * 78}\n")
    print(
        f"  {'Model':<18}{'Promoting':>14}{'Jaccard':>10}"
        f"{'Suppressing':>16}{'Jaccard':>10}"
    )
    print(f"  {'-' * 68}")
    for model, r in recovery.items():
        prom = f"{r['promoting_recovered']}/{r['n_promoting_color']}"
        supp = f"{r['suppressing_recovered']}/{r['n_suppressing_color']}"
        print(
            f"  {model:<18}{prom:>14}{r['jaccard_promoting']:>10.2f}"
            f"{supp:>16}{r['jaccard_suppressing']:>10.2f}"
        )


def print_ablation(summary: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print("  KNOCKOUT ABLATION ON WHOOPS")
    print(f"{'=' * 78}\n")
    for knockout_id in GROUP_KNOCKOUTS:
        print(f"  {knockout_id}")
        print(
            f"    {'Model':<18}{'n':>5}{'Prior flip':>13}"
            f"{'Visual flip':>13}{'McNemar p':>12}"
        )
        print(f"    {'-' * 61}")
        for model, configs in summary.items():
            if knockout_id not in configs:
                continue
            s = configs[knockout_id]
            print(
                f"    {model:<18}{s['n']:>5}{s['prior_flip_rate']:>12.1%}"
                f"{s['visual_flip_rate']:>12.1%}{_p_str(s['mcnemar_p']):>12}"
            )
        print()


def print_latex(recovery: dict[str, dict], summary: dict[str, dict]) -> None:
    """Emit the section 4.7 table body, ready to paste into the paper."""
    print(f"\n{'=' * 78}")
    print("  LATEX (section 4.7 table body)")
    print(f"{'=' * 78}\n")
    for model, r in recovery.items():
        s = summary.get(model, {}).get("grp_promoting")
        if s is None:
            continue
        print(
            f"{model} & {s['n']} & "
            f"{r['promoting_recovered']}/{r['n_promoting_color']} & "
            f"{r['suppressing_recovered']}/{r['n_suppressing_color']} & "
            f"{s['prior_flip_rate']:.1%} & {s['visual_flip_rate']:.1%} & "
            f"{_p_str(s['mcnemar_p'])} \\\\".replace("%", r"\%")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latex", action="store_true", help="Also emit the LaTeX table body."
    )
    args = parser.parse_args()

    with COLOR_CLASSIFICATIONS.open() as f:
        color = json.load(f)
    with WHOOPS_CLASSIFICATIONS.open() as f:
        whoops = json.load(f)
    recovery = compute_recovery(color, whoops)

    print(f"Loading WHOOPS knockout ablation results from {KNOCKOUT_DIR} ...")
    results = load_whoops_knockout_results(KNOCKOUT_DIR)
    allowed = load_patching_example_ids(PATCHING_DIR)
    kept = restrict_to_classified(results, allowed)
    print(
        f"Loaded {len(results)} results across {len(recovery)} models; "
        f"{len(kept)} remain after restricting to classified examples."
    )
    summary = compute_ablation_summary(kept)

    print_recovery(recovery)
    print_ablation(summary)
    if args.latex:
        print_latex(recovery, summary)


if __name__ == "__main__":
    main()
