import json
import statistics
from pathlib import Path


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Change this if your JSON has a different name.
INPUT_FILE = "head_screening_results.json"


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("HEAD SCREENING ANALYSIS")
print("=" * 70)

print(f"Examples loaded: {len(data)}")


# ---------------------------------------------------------
# COLLECT HEAD DATA
# ---------------------------------------------------------

head_data = {}

for example in data:

    heads = example.get("heads", {})

    for head_name, values in heads.items():

        if head_name not in head_data:
            head_data[head_name] = {
                "prior_norm": [],
                "visual_norm": [],
                "difference_norm": [],
                "cosine": [],
                "causal_effect": [],
            }

        head_data[head_name]["prior_norm"].append(
            values["prior_norm"]
        )

        head_data[head_name]["visual_norm"].append(
            values["visual_norm"]
        )

        head_data[head_name]["difference_norm"].append(
            values["difference_norm"]
        )

        head_data[head_name]["cosine"].append(
            values["cosine"]
        )

        # Causal effect may not exist in this JSON.
        if "causal_effect" in values:
            head_data[head_name]["causal_effect"].append(
                values["causal_effect"]
            )


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

def mean(values):
    return statistics.mean(values) if values else None


def std(values):
    return statistics.stdev(values) if len(values) > 1 else None


def coefficient_of_variation(values):
    if len(values) < 2:
        return None

    m = mean(values)

    if m == 0:
        return None

    return std(values) / abs(m)


# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("HEAD SUMMARY")
print("=" * 70)

for head, metrics in head_data.items():

    print(f"\n{head}")
    print("-" * 50)

    for metric in [
        "prior_norm",
        "visual_norm",
        "difference_norm",
        "cosine",
        "causal_effect",
    ]:

        values = metrics[metric]

        if not values:
            continue

        print(
            f"{metric:18s} "
            f"mean={mean(values):8.4f} "
            f"std={std(values):8.4f}"
        )


# ---------------------------------------------------------
# RANKINGS
# ---------------------------------------------------------

def ranking(metric, reverse=True):

    available = []

    for head, metrics in head_data.items():

        values = metrics[metric]

        if values:
            available.append(
                (head, mean(values))
            )

    return sorted(
        available,
        key=lambda x: x[1],
        reverse=reverse
    )


print("\n")
print("=" * 70)
print("RANKING: DIFFERENCE NORM")
print("=" * 70)

for i, (head, value) in enumerate(
    ranking("difference_norm", reverse=True),
    start=1
):
    print(f"{i:2d}. {head:10s} {value:.4f}")


print("\n")
print("=" * 70)
print("RANKING: COSINE SIMILARITY")
print("=" * 70)

for i, (head, value) in enumerate(
    ranking("cosine", reverse=False),
    start=1
):
    print(f"{i:2d}. {head:10s} {value:.4f}")


print("\n")
print("=" * 70)
print("RANKING: VISUAL NORM")
print("=" * 70)

for i, (head, value) in enumerate(
    ranking("visual_norm", reverse=True),
    start=1
):
    print(f"{i:2d}. {head:10s} {value:.4f}")


print("\n")
print("=" * 70)
print("RANKING: PRIOR NORM")
print("=" * 70)

for i, (head, value) in enumerate(
    ranking("prior_norm", reverse=True),
    start=1
):
    print(f"{i:2d}. {head:10s} {value:.4f}")


# ---------------------------------------------------------
# VARIABILITY
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("HEAD VARIABILITY")
print("=" * 70)

variability = []

for head, metrics in head_data.items():

    values = metrics["difference_norm"]

    if len(values) > 1:

        cv = coefficient_of_variation(values)

        variability.append(
            (head, cv, std(values), mean(values))
        )

variability.sort(
    key=lambda x: x[1],
    reverse=True
)

for i, (head, cv, s, m) in enumerate(
    variability,
    start=1
):

    print(
        f"{i:2d}. {head:10s} "
        f"CV={cv:.4f} "
        f"std={s:.4f} "
        f"mean={m:.4f}"
    )


# ---------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------

summary = {}

for head, metrics in head_data.items():

    summary[head] = {}

    for metric, values in metrics.items():

        if values:

            summary[head][metric] = {
                "mean": mean(values),
                "std": std(values),
                "min": min(values),
                "max": max(values),
            }


OUTPUT_FILE = "head_analysis_summary.json"

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(f"Saved summary to: {OUTPUT_FILE}")