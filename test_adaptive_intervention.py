import torch

from vdpo.adaptive_gate import AdaptiveHeadGate
from vdpo.gate_features import (
    load_gate_features,
    PROMOTING_HEADS,
)
from vdpo.interventions import adaptive_head_weighting

from vdpo.types.enums import (
    GroundingMode,
    ModelFamily,
    ModelSize,
)

from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


def get_logits(model, inputs):

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[0, -1, :].detach().cpu()


def main():

    print("=" * 70)
    print("ADAPTIVE HEAD INTERVENTION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load gate features
    # ---------------------------------------------------------

    features, metadata = load_gate_features(
        "head_screening_results.json"
    )

    print()
    print("Feature shape:", tuple(features.shape))

    # ---------------------------------------------------------
    # Create gate
    # ---------------------------------------------------------

    gate = AdaptiveHeadGate(
        input_dim=features.shape[1],
        num_heads=len(PROMOTING_HEADS),
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print()
    print("Loading Qwen2.5-VL 3B...")

    loaded = load_vlm(
        model_size=ModelSize.SMALL_3B,
        model_family=ModelFamily.QWEN,
    )

    model = loaded.model
    processor = loaded.processor
    adapter = loaded.adapter

    print("Model loaded.")

    # ---------------------------------------------------------
    # Load examples
    # ---------------------------------------------------------

    examples = load_color_examples()[:20]

    # ---------------------------------------------------------
    # Run examples
    # ---------------------------------------------------------

    for idx, (example, feature_vector) in enumerate(
        zip(examples, features),
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"Example {idx}/20: "
            f"{example.object}"
        )
        print("=" * 70)

        print(
            "Prior color:",
            example.correct_answer[0],
        )

        print(
            "Visual color:",
            example.incorrect_answer,
        )

        # -----------------------------------------------------
        # Generate adaptive weights
        # -----------------------------------------------------

        with torch.no_grad():

            weights = gate(
                feature_vector.unsqueeze(0)
            )[0]

        print()
        print("Adaptive weights:")

        for head, weight in zip(
            PROMOTING_HEADS,
            weights,
        ):

            print(
                f"L{head[0]}_H{head[1]}: "
                f"{weight.item():.4f}"
            )

        # -----------------------------------------------------
        # Build prior prompt
        # -----------------------------------------------------

        question = VMIRunnerBase._get_question(
            example,
            GroundingMode.PRIOR,
        )

        image = example.counterfact_image.pil_image

        inputs = adapter.build_inputs(
            processor=processor,
            image=image,
            question=question,
            device=model.device,
        )

        # -----------------------------------------------------
        # Candidate token IDs
        # -----------------------------------------------------

        prior_color = example.correct_answer[0]
        visual_color = example.incorrect_answer

        prior_ids = processor.tokenizer(
            prior_color,
            add_special_tokens=False,
        ).input_ids

        visual_ids = processor.tokenizer(
            visual_color,
            add_special_tokens=False,
        ).input_ids

        prior_id = prior_ids[0]
        visual_id = visual_ids[0]

        # -----------------------------------------------------
        # BASELINE
        # -----------------------------------------------------

        baseline = get_logits(
            model,
            inputs,
        )

        baseline_prior = baseline[prior_id].item()
        baseline_visual = baseline[visual_id].item()

        # -----------------------------------------------------
        # ADAPTIVE INTERVENTION
        # -----------------------------------------------------

        with adaptive_head_weighting(
            model,
            adapter,
            PROMOTING_HEADS,
            weights,
        ):

            adaptive_logits = get_logits(
                model,
                inputs,
            )

        adaptive_prior = adaptive_logits[
            prior_id
        ].item()

        adaptive_visual = adaptive_logits[
            visual_id
        ].item()

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        baseline_margin = (
            baseline_visual -
            baseline_prior
        )

        adaptive_margin = (
            adaptive_visual -
            adaptive_prior
        )

        print()
        print("-" * 70)

        print("BASELINE")
        print(
            f"{prior_color}: "
            f"{baseline_prior:.4f}"
        )

        print(
            f"{visual_color}: "
            f"{baseline_visual:.4f}"
        )

        print(
            f"Visual - Prior: "
            f"{baseline_margin:.4f}"
        )

        print()
        print("ADAPTIVE")
        print(
            f"{prior_color}: "
            f"{adaptive_prior:.4f}"
        )

        print(
            f"{visual_color}: "
            f"{adaptive_visual:.4f}"
        )

        print(
            f"Visual - Prior: "
            f"{adaptive_margin:.4f}"
        )

        print()
        print(
            "Margin change:",
            f"{adaptive_margin - baseline_margin:.4f}"
        )

    print()
    print("=" * 70)
    print("ADAPTIVE INTERVENTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()