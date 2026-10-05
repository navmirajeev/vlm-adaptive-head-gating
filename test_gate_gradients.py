import torch

from vdpo.adaptive_gate import AdaptiveHeadGate
from vdpo.gate_features import (
    load_gate_features,
    PROMOTING_HEADS,
)

from vdpo.interventions import (
    adaptive_head_weighting_trainable,
)

from vdpo.types.enums import (
    GroundingMode,
    ModelFamily,
    ModelSize,
)

from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


def main():

    print("=" * 70)
    print("ADAPTIVE GATE GRADIENT TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load features
    # ---------------------------------------------------------

    features, metadata = load_gate_features(
        "head_screening_results.json"
    )

    # Use only the first example for this test.

    feature_vector = features[0].unsqueeze(0)

    print()
    print("Feature shape:", tuple(feature_vector.shape))

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
    # Load first example
    # ---------------------------------------------------------

    example = load_color_examples()[0]

    print()
    print("Example:", example.object)

    prior_color = example.correct_answer[0]
    visual_color = example.incorrect_answer

    print("Prior color:", prior_color)
    print("Visual color:", visual_color)

    # ---------------------------------------------------------
    # Build input
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Generate weights
    # ---------------------------------------------------------

    weights = gate(feature_vector)[0]

    print()
    print("Initial weights:")

    for head, weight in zip(
        PROMOTING_HEADS,
        weights,
    ):

        print(
            f"L{head[0]}_H{head[1]}: "
            f"{weight.item():.4f}"
        )

    # ---------------------------------------------------------
    # Token IDs
    # ---------------------------------------------------------

    prior_id = processor.tokenizer(
        prior_color,
        add_special_tokens=False,
    ).input_ids[0]

    visual_id = processor.tokenizer(
        visual_color,
        add_special_tokens=False,
    ).input_ids[0]

    # ---------------------------------------------------------
    # Adaptive forward pass
    # ---------------------------------------------------------

    with adaptive_head_weighting_trainable(
        model,
        adapter,
        PROMOTING_HEADS,
        weights,
    ):

        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    prior_logit = logits[prior_id]

    visual_logit = logits[visual_id]

    # ---------------------------------------------------------
    # Simple development loss
    # ---------------------------------------------------------

    margin = visual_logit - prior_logit

    loss = -margin

    print()
    print("Visual logit:", visual_logit.item())
    print("Prior logit:", prior_logit.item())
    print("Margin:", margin.item())
    print("Loss:", loss.item())

    # ---------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------

    gate.zero_grad()

    loss.backward()

    print()
    print("Gradient check:")
    print("-" * 50)

    total_gradient = 0.0

    for name, parameter in gate.named_parameters():

        if parameter.grad is None:

            print(
                f"{name}: NO GRADIENT"
            )

        else:

            gradient_size = (
                parameter.grad.abs().sum().item()
            )

            total_gradient += gradient_size

            print(
                f"{name}: "
                f"{gradient_size:.8f}"
            )

    print()
    print(
        "Total gate gradient:",
        f"{total_gradient:.8f}",
    )

    print()
    print("=" * 70)

    if total_gradient > 0:

        print("SUCCESS: gradients reach the gate.")

    else:

        print("FAILURE: gate received no gradients.")

    print("=" * 70)


if __name__ == "__main__":
    main()