import json
from pathlib import Path

import pytest

from scripts.analysis.whoops_summary import (
    compute_ablation_summary,
    compute_recovery,
    head_set,
    load_patching_example_ids,
    load_whoops_knockout_results,
    mcnemar_exact,
    restrict_to_classified,
)


def _write_knockout(
    knockout_dir: Path,
    family: str,
    size: str,
    example_id: str,
    knockout_id: str,
    visual_predicted: str,
    prior_predicted: str,
):
    p = knockout_dir / family / size / f"{example_id}__{knockout_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "example_id": example_id,
                "knockout_id": knockout_id,
                "visual_grounding": {"predicted": visual_predicted},
                "prior_grounding": {"predicted": prior_predicted},
            }
        )
    )


# --- head_set -------------------------------------------------------------


def test_head_set_normalizes_layer_head_dicts():
    entries = [{"layer": 24, "head": 6}, {"layer": 31, "head": 7}]
    assert head_set(entries) == {(24, 6), (31, 7)}


def test_head_set_of_empty_list_is_empty():
    assert head_set([]) == set()


# --- mcnemar_exact --------------------------------------------------------


def test_mcnemar_reproduces_published_qwen_3b_value():
    """15 prior-only flips vs 3 visual-only flips is the published p = .0075."""
    assert mcnemar_exact(15, 3) == pytest.approx(0.0075, abs=5e-5)


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(15, 3) == pytest.approx(mcnemar_exact(3, 15))


def test_mcnemar_with_no_discordant_pairs_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_clamped_to_one():
    """b == c doubles a probability above 0.5; the result must stay a p-value."""
    assert mcnemar_exact(1, 1) == 1.0


def test_mcnemar_one_sided_extreme():
    """5 discordant pairs all in one direction: 2 * (1/2)^5."""
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)


# --- compute_recovery -----------------------------------------------------


def test_recovery_counts_color_heads_rediscovered_on_whoops():
    color = {
        "qwen/3B": {
            "promoting": [{"layer": 1, "head": 0}, {"layer": 2, "head": 0}],
            "suppressing": [{"layer": 3, "head": 0}],
        }
    }
    whoops = {
        "qwen/3B": {
            # Recovers L1H0, misses L2H0, and adds L9H0 of its own.
            "promoting": [{"layer": 1, "head": 0}, {"layer": 9, "head": 0}],
            "suppressing": [{"layer": 3, "head": 0}],
        }
    }
    rec = compute_recovery(color, whoops)["qwen/3B"]

    assert rec["n_promoting_color"] == 2
    assert rec["n_promoting_whoops"] == 2
    assert rec["promoting_recovered"] == 1
    assert rec["jaccard_promoting"] == pytest.approx(1 / 3)
    assert rec["suppressing_recovered"] == 1
    assert rec["jaccard_suppressing"] == pytest.approx(1.0)


def test_recovery_skips_models_missing_from_whoops():
    color = {"qwen/3B": {"promoting": [], "suppressing": []}}
    assert compute_recovery(color, {}) == {}


# --- compute_ablation_summary ---------------------------------------------


def _results(pairs, knockout_id="grp_promoting", model=("qwen", "3B")):
    """pairs: list of (visual_predicted, prior_predicted)."""
    return [
        {
            "example_id": f"whoops_{i:04d}",
            "knockout_id": knockout_id,
            "model_family": model[0],
            "model_size": model[1],
            "visual_grounding": {"predicted": v},
            "prior_grounding": {"predicted": p},
        }
        for i, (v, p) in enumerate(pairs)
    ]


def test_ablation_summary_computes_flip_rates_per_model():
    # 2 of 4 prior flips, 1 of 4 visual flips.
    results = _results(
        [
            ("visual", "visual"),  # prior flipped
            ("visual", "visual"),  # prior flipped
            ("prior", "prior"),  # visual flipped
            ("visual", "prior"),  # neither flipped
        ]
    )
    s = compute_ablation_summary(results)["qwen/3B"]["grp_promoting"]

    assert s["n"] == 4
    assert s["prior_flips"] == 2
    assert s["visual_flips"] == 1
    assert s["prior_flip_rate"] == pytest.approx(0.5)
    assert s["visual_flip_rate"] == pytest.approx(0.25)


def test_ablation_summary_counts_discordant_pairs_for_mcnemar():
    results = _results(
        [
            ("visual", "visual"),  # prior only  -> b
            ("visual", "visual"),  # prior only  -> b
            ("prior", "prior"),  # visual only -> c
            ("prior", "visual"),  # both flipped, concordant
            ("visual", "prior"),  # neither flipped, concordant
        ]
    )
    s = compute_ablation_summary(results)["qwen/3B"]["grp_promoting"]

    assert s["prior_only"] == 2
    assert s["visual_only"] == 1
    assert s["mcnemar_p"] == pytest.approx(mcnemar_exact(2, 1))


def test_ablation_summary_separates_models_and_knockout_ids():
    results = _results([("visual", "visual")], knockout_id="grp_promoting")
    results += _results([("prior", "prior")], knockout_id="grp_suppressing")
    results += _results(
        [("visual", "visual")],
        knockout_id="grp_promoting",
        model=("paligemma_2", "10B"),
    )
    summary = compute_ablation_summary(results)

    assert set(summary) == {"qwen/3B", "paligemma_2/10B"}
    assert set(summary["qwen/3B"]) == {"grp_promoting", "grp_suppressing"}
    assert summary["paligemma_2/10B"]["grp_promoting"]["n"] == 1


# --- load_whoops_knockout_results -----------------------------------------


def test_loader_reads_family_and_size_from_path(tmp_path):
    d = tmp_path / "knockout_attn_heads"
    _write_knockout(d, "qwen", "3B", "whoops_0001", "grp_promoting", "visual", "visual")
    _write_knockout(
        d, "paligemma_2", "10B", "whoops_0002", "grp_promoting", "prior", "prior"
    )

    results = load_whoops_knockout_results(d)

    assert {(r["model_family"], r["model_size"]) for r in results} == {
        ("qwen", "3B"),
        ("paligemma_2", "10B"),
    }


def test_loader_excludes_frozen_color_side_analysis(tmp_path):
    """outputs/whoops/knockout_attn_heads/frozen_color/ is an unreported side run.

    It nests one level deeper (frozen_color/qwen/3B), so without an explicit
    exclusion it would be read as model family "frozen_color", size "qwen".
    """
    d = tmp_path / "knockout_attn_heads"
    _write_knockout(d, "qwen", "3B", "whoops_0001", "grp_promoting", "visual", "visual")
    _write_knockout(
        d / "frozen_color",
        "qwen",
        "3B",
        "whoops_0001",
        "grp_promoting",
        "prior",
        "prior",
    )

    results = load_whoops_knockout_results(d)

    assert len(results) == 1
    assert results[0]["model_family"] == "qwen"


def test_loader_applies_no_correctness_filter(tmp_path):
    """The whoops subset is already flip-filtered upstream, so nothing is dropped."""
    d = tmp_path / "knockout_attn_heads"
    for i in range(3):
        _write_knockout(
            d, "qwen", "3B", f"whoops_{i:04d}", "grp_promoting", "visual", "prior"
        )

    assert len(load_whoops_knockout_results(d)) == 3


# --- restrict_to_classified ------------------------------------------------


def test_patching_example_ids_are_grouped_by_model(tmp_path):
    d = tmp_path / "patching"
    for family, size, eid in [
        ("qwen", "3B", "whoops_0001"),
        ("qwen", "3B", "whoops_0002"),
        ("llava_next", "7B", "whoops_0003"),
    ]:
        p = d / family / size / f"{eid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")

    assert load_patching_example_ids(d) == {
        "qwen/3B": {"whoops_0001", "whoops_0002"},
        "llava_next/7B": {"whoops_0003"},
    }


def test_restrict_drops_ablation_examples_without_patching_data():
    """An example can have ablation data but no patching data, so it never

    entered the classification. Keeping it would evaluate the circuit on an
    example that had no chance to shape it.
    """
    results = _results([("visual", "visual"), ("prior", "prior")])
    allowed = {"qwen/3B": {"whoops_0000"}}

    kept = restrict_to_classified(results, allowed)

    assert [r["example_id"] for r in kept] == ["whoops_0000"]


def test_restrict_drops_all_examples_for_a_model_with_no_patching_data():
    results = _results([("visual", "visual")])
    assert restrict_to_classified(results, {}) == []
