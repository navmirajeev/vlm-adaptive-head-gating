import torch

from vdpo.representation_gates import HeadContextGate
from vdpo.representation_intervention import adaptive_head_context_gate

from vdpo.types.enums import (
    GroundingMode,
    ModelFamily,
    ModelSize,
)

from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


PROMOTING_HEADS = [
    (26, 0),
    (26, 5),
    (26, 6),
    (27, 1),
    (27, 4),
    (28, 3),
    (31, 3),
    (31, 7),
    (34, 14),
]


def get_logits(model, inputs):

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[0, -1, :].detach().cpu()


def main():

    print("=" * 70)
    print("OPTION 2: HEAD + SAME-TOKEN CONTEXT TEST")
    print("=" * 70)

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

    # Freeze Qwen
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    # ---------------------------------------------------------
    # Create Option 2 context projector
    # ---------------------------------------------------------

    context_projector = torch.nn.Sequential(
        torch.nn.Linear(
            2048,
            32,
        ),
        torch.nn.GELU(),
    ).to(model.device)

    # ---------------------------------------------------------
    # Create one HeadContextGate per selected head
    # ---------------------------------------------------------

    gates = {}

    for head in PROMOTING_HEADS:

        gates[head] = HeadContextGate(
            head_dim=128,
            context_dim=32,
            hidden_dim=8,
        ).to(model.device)

    print()
    print(
        f"Created {len(gates)} contextual head gates."
    )

    print(
        "Context projector: 2048 -> 32"
    )

    # ---------------------------------------------------------
    # Load examples
    # ---------------------------------------------------------

    examples = load_color_examples()[:2]

    # ---------------------------------------------------------
    # Test examples
    # ---------------------------------------------------------

    for idx, example in enumerate(
        examples,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"Example {idx}/{len(examples)}: "
            f"{example.object}"
        )
        print("=" * 70)

        prior_color = example.correct_answer[0]
        visual_color = example.incorrect_answer

        print(
            "Prior color:",
            prior_color,
        )

        print(
            "Visual color:",
            visual_color,
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

        print()
        print("Running baseline forward pass...")

        baseline = get_logits(
            model,
            inputs,
        )

        baseline_prior = baseline[
            prior_id
        ].item()

        baseline_visual = baseline[
            visual_id
        ].item()

        baseline_margin = (
            baseline_visual -
            baseline_prior
        )

        print(
            f"Baseline {prior_color}: "
            f"{baseline_prior:.4f}"
        )

        print(
            f"Baseline {visual_color}: "
            f"{baseline_visual:.4f}"
        )

        print(
            f"Baseline Visual - Prior: "
            f"{baseline_margin:.4f}"
        )

        # -----------------------------------------------------
        # OPTION 2 INTERVENTION
        # -----------------------------------------------------

        print()
        print(
            "Running Head + Context intervention..."
        )

        with adaptive_head_context_gate(
            model=model,
            adapter=adapter,
            heads=PROMOTING_HEADS,
            gates=gates,
            context_projector=context_projector,
        ):

            contextual = get_logits(
                model,
                inputs,
            )

        contextual_prior = contextual[
            prior_id
        ].item()

        contextual_visual = contextual[
            visual_id
        ].item()

        contextual_margin = (
            contextual_visual -
            contextual_prior
        )

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        print()
        print("-" * 70)

        print("OPTION 2")

        print(
            f"{prior_color}: "
            f"{contextual_prior:.4f}"
        )

        print(
            f"{visual_color}: "
            f"{contextual_visual:.4f}"
        )

        print(
            f"Visual - Prior: "
            f"{contextual_margin:.4f}"
        )

        print()
        print(
            "Margin change:",
            f"{contextual_margin - baseline_margin:.4f}"
        )

        print(
            "Prior logit change:",
            f"{contextual_prior - baseline_prior:.4f}"
        )

        print(
            "Visual logit change:",
            f"{contextual_visual - baseline_visual:.4f}"
        )

    print()
    print("=" * 70)
    print("OPTION 2 FORWARD TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()