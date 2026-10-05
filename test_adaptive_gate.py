import torch

from vdpo.adaptive_gate import AdaptiveHeadGate
from vdpo.gate_features import (
    load_gate_features,
    PROMOTING_HEADS,
)


def main():

    print("=" * 70)
    print("ADAPTIVE HEAD GATE - REAL FEATURE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load existing screening results
    # ---------------------------------------------------------

    features, metadata = load_gate_features(
        "head_screening_results.json"
    )

    print()
    print("Feature tensor shape:", tuple(features.shape))

    # Expected:
    #
    # 20 examples × 36 features

    # ---------------------------------------------------------
    # Create gate
    # ---------------------------------------------------------

    gate = AdaptiveHeadGate(
        input_dim=features.shape[1],
        num_heads=len(PROMOTING_HEADS),
    )

    # ---------------------------------------------------------
    # Generate adaptive weights
    # ---------------------------------------------------------

    with torch.no_grad():
        weights = gate(features)

    print()
    print("Generated weights shape:")
    print(tuple(weights.shape))

    # ---------------------------------------------------------
    # Show every example
    # ---------------------------------------------------------

    for example_idx, (example_weights, info) in enumerate(
        zip(weights, metadata),
        start=1,
    ):

        print()
        print("-" * 70)

        print(
            f"Example {example_idx}: "
            f"{info['object']}"
        )

        print(
            f"Prior color:  {info['prior_color']}"
        )

        print(
            f"Visual color: {info['visual_color']}"
        )

        print()

        for head, weight in zip(
            PROMOTING_HEADS,
            example_weights,
        ):

            print(
                f"L{head[0]}_H{head[1]}: "
                f"{weight.item():.4f}"
            )

    # ---------------------------------------------------------
    # Global statistics
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("GATE OUTPUT STATISTICS")
    print("=" * 70)

    print(
        f"Minimum weight: "
        f"{weights.min().item():.4f}"
    )

    print(
        f"Maximum weight: "
        f"{weights.max().item():.4f}"
    )

    print(
        f"Mean weight: "
        f"{weights.mean().item():.4f}"
    )

    # ---------------------------------------------------------
    # Mean weight per head
    # ---------------------------------------------------------

    print()
    print("Mean weight per head:")
    print("-" * 50)

    mean_weights = weights.mean(dim=0)

    for head, weight in zip(
        PROMOTING_HEADS,
        mean_weights,
    ):

        print(
            f"L{head[0]}_H{head[1]}: "
            f"{weight.item():.4f}"
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()