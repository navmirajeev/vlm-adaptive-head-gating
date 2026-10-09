import csv
import os

import torch

from vdpo.representation_gates import (
    HeadOnlyGate,
    HeadContextGate,
)
from vdpo.representation_intervention import (
    adaptive_head_gate,
    adaptive_head_context_gate,
)
from vdpo.types.enums import (
    GroundingMode,
    ModelFamily,
    ModelSize,
)
from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

NUM_EXAMPLES = 20

OPTION1_FILE = "trained_head_only_gates.pt"
OPTION2_FILE = "trained_head_context_gates.pt"

OUTPUT_CSV = "gating_comparison_results.csv"


# ---------------------------------------------------------
# Input preparation
# ---------------------------------------------------------

def get_example_inputs(example, model, processor, adapter):
    question = VMIRunnerBase._get_question(
        example,
        GroundingMode.PRIOR,
    )

    image = example.counterfact_image.pil_image

    return adapter.build_inputs(
        processor=processor,
        image=image,
        question=question,
        device=model.device,
    )


def get_candidate_ids(example, processor):
    prior_ids = processor.tokenizer(
        example.correct_answer[0],
        add_special_tokens=False,
    ).input_ids

    visual_ids = processor.tokenizer(
        example.incorrect_answer,
        add_special_tokens=False,
    ).input_ids

    if not prior_ids or not visual_ids:
        raise ValueError(
            f"Empty candidate tokenization for {example.object}"
        )

    return prior_ids[0], visual_ids[0]


# ---------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------

def load_option1(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Option 1 checkpoint not found: {path}"
        )

    saved = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    gates = {}

    for layer, head in PROMOTING_HEADS:
        key = f"L{layer}_H{head}"

        gate = HeadOnlyGate(
            head_dim=128,
            hidden_dim=8,
        ).to(device=device, dtype=torch.float32)

        if key not in saved["gates"]:
            raise KeyError(
                f"{key} missing from Option 1 checkpoint"
            )

        gate.load_state_dict(saved["gates"][key])
        gate.eval()
        gates[(layer, head)] = gate

    return gates


def load_option2(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Option 2 checkpoint not found: {path}"
        )

    saved = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    gates = {}

    for layer, head in PROMOTING_HEADS:
        key = f"L{layer}_H{head}"

        gate = HeadContextGate(
            head_dim=128,
            context_dim=32,
            hidden_dim=8,
        ).to(device=device, dtype=torch.float32)

        if key not in saved["gates"]:
            raise KeyError(
                f"{key} missing from Option 2 checkpoint"
            )

        gate.load_state_dict(saved["gates"][key])
        gate.eval()
        gates[(layer, head)] = gate

    context_projector = torch.nn.Sequential(
        torch.nn.Linear(2048, 32),
        torch.nn.GELU(),
    ).to(device=device, dtype=torch.float32)

    context_projector.load_state_dict(
        saved["context_projector"]
    )
    context_projector.eval()

    return gates, context_projector


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def run_condition(
    condition,
    model,
    adapter,
    inputs,
    prior_id,
    visual_id,
    gates=None,
    context_projector=None,
):
    if condition == "Baseline":
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]

    elif condition == "Option 1":
        with torch.no_grad():
            with adaptive_head_gate(
                model=model,
                adapter=adapter,
                heads=PROMOTING_HEADS,
                gates=gates,
            ):
                logits = model(**inputs).logits[0, -1, :]

    elif condition == "Option 2":
        with torch.no_grad():
            with adaptive_head_context_gate(
                model=model,
                adapter=adapter,
                heads=PROMOTING_HEADS,
                gates=gates,
                context_projector=context_projector,
            ):
                logits = model(**inputs).logits[0, -1, :]

    else:
        raise ValueError(f"Unknown condition: {condition}")

    prior_logit = logits[prior_id].item()
    visual_logit = logits[visual_id].item()
    margin = visual_logit - prior_logit

    # This is a two-candidate comparison, not full-vocabulary accuracy.
    visual_wins = visual_logit > prior_logit

    return {
        "prior_logit": prior_logit,
        "visual_logit": visual_logit,
        "visual_minus_prior": margin,
        "visual_candidate_wins": int(visual_wins),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("BASELINE VS ADAPTIVE GATING COMPARISON")
    print("=" * 70)

    # Load Qwen
    print("\nLoading Qwen2.5-VL 3B...")

    loaded = load_vlm(
        model_size=ModelSize.SMALL_3B,
        model_family=ModelFamily.QWEN,
    )

    model = loaded.model
    processor = loaded.processor
    adapter = loaded.adapter

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    device = model.device

    print("Model loaded.")
    print("Qwen parameters frozen.")

    # Load both trained gate sets
    print("\nLoading Option 1 checkpoint...")
    option1_gates = load_option1(OPTION1_FILE, device)

    print("Loading Option 2 checkpoint...")
    option2_gates, context_projector = load_option2(
        OPTION2_FILE,
        device,
    )

    print("Both checkpoints loaded.")

    # Use exactly the same examples in each condition.
    examples = load_color_examples()[:NUM_EXAMPLES]

    conditions = [
        ("Baseline", None, None),
        ("Option 1", option1_gates, None),
        ("Option 2", option2_gates, context_projector),
    ]

    results = []

    print(f"\nEvaluating {len(examples)} examples")
    print("Each example is tested under all three conditions.")

    for index, example in enumerate(examples, start=1):
        print(
            f"\n[{index}/{len(examples)}] "
            f"{example.object}"
        )

        inputs = get_example_inputs(
            example,
            model,
            processor,
            adapter,
        )

        prior_id, visual_id = get_candidate_ids(
            example,
            processor,
        )

        for condition, gates, projector in conditions:
            scores = run_condition(
                condition=condition,
                model=model,
                adapter=adapter,
                inputs=inputs,
                prior_id=prior_id,
                visual_id=visual_id,
                gates=gates,
                context_projector=projector,
            )

            row = {
                "example_index": index,
                "object": example.object,
                "prior_answer": example.correct_answer[0],
                "visual_answer": example.incorrect_answer,
                "condition": condition,
                **scores,
            }

            results.append(row)

            print(
                f"  {condition:10s} | "
                f"margin={scores['visual_minus_prior']:.4f} | "
                f"visual wins={scores['visual_candidate_wins']}"
            )

    # Save individual example results.
    fieldnames = [
        "example_index",
        "object",
        "prior_answer",
        "visual_answer",
        "condition",
        "prior_logit",
        "visual_logit",
        "visual_minus_prior",
        "visual_candidate_wins",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    # Summarize margins by condition.
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for condition, _, _ in conditions:
        subset = [
            row for row in results
            if row["condition"] == condition
        ]

        mean_margin = sum(
            row["visual_minus_prior"] for row in subset
        ) / len(subset)

        visual_wins = sum(
            row["visual_candidate_wins"] for row in subset
        )

        print(f"\n{condition}")
        print(f"  Examples: {len(subset)}")
        print(f"  Mean visual-minus-prior margin: {mean_margin:.4f}")
        print(f"  Visual candidate wins: {visual_wins}/{len(subset)}")

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()