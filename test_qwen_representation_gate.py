import torch

from vdpo.representation_gates import HeadOnlyGate
from vdpo.representation_intervention import adaptive_head_gate

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
    print("QWEN REPRESENTATION-BASED HEAD GATE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load Qwen
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
    # Freeze Qwen
    # ---------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("Qwen parameters frozen.")

    # ---------------------------------------------------------
    # Create one gate per selected head
    # ---------------------------------------------------------

    gates = {}

    for head in PROMOTING_HEADS:

        gate = HeadOnlyGate(
            head_dim=128,
            hidden_dim=8,
        )

        gate.to(model.device)

        gates[head] = gate

    print()
    print(
        "Created",
        len(gates),
        "independent head gates."
    )

    # ---------------------------------------------------------
    # Count gate parameters
    # ---------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for gate in gates.values()
        for parameter in gate.parameters()
    )

    print(
        "Trainable gate parameters:",
        total_parameters,
    )

    # 9 heads × 1041 parameters
    #
    # = 9369 parameters

    # ---------------------------------------------------------
    # Load example
    # ---------------------------------------------------------

    examples = load_color_examples()

    example = examples[0]

    image = example.counterfact_image.pil_image

    question = VMIRunnerBase._get_question(
        example,
        GroundingMode.PRIOR,
    )

    inputs = adapter.build_inputs(
        processor=processor,
        image=image,
        question=question,
        device=model.device,
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline = get_logits(
        model,
        inputs,
    )

    # ---------------------------------------------------------
    # Adaptive gate
    # ---------------------------------------------------------

    with adaptive_head_gate(
        model,
        adapter,
        PROMOTING_HEADS,
        gates,
    ):

        gated_logits = get_logits(
            model,
            inputs,
        )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    difference = (
        gated_logits - baseline
    )

    print()
    print("=" * 70)
    print("BASELINE VS ADAPTIVE GATE")
    print("=" * 70)

    print(
        "Maximum logit change:",
        difference.abs().max().item(),
    )

    print(
        "Mean absolute logit change:",
        difference.abs().mean().item(),
    )

    # ---------------------------------------------------------
    # Show gate values
    # ---------------------------------------------------------

    print()
    print("Gate outputs:")

    print("-" * 50)

    for head, gate in gates.items():

        # We cannot directly obtain the gate value here
        # without another forward pass, so this section
        # simply confirms the gate exists.

        print(
            f"L{head[0]}_H{head[1]}: "
            f"ready"
        )

    # ---------------------------------------------------------
    # Check parameters
    # ---------------------------------------------------------

    trainable_model_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print()
    print(
        "Trainable Qwen parameters:",
        sum(
            p.numel()
            for p in trainable_model_parameters
        ),
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()