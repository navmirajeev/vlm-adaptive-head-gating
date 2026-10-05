import torch

from vdpo.interventions import head_weighting
from vdpo.types.enums import GroundingMode, ModelFamily, ModelSize
from vdpo.utils.load_examples import load_color_examples
from vdpo.utils.load_model import load_vlm
from vdpo.runner.core import VMIRunnerBase


PROMOTING_HEADS = [
    (26, 0), (26, 5), (26, 6),
    (27, 1), (27, 4),
    (28, 3),
    (31, 3), (31, 7),
    (34, 14),
]


@torch.no_grad()
def get_logits(model, inputs):
    outputs = model(**inputs)
    return outputs.logits[0, -1, :].detach().cpu()


def main():
    print("Loading Qwen2.5-VL 3B...")

    loaded = load_vlm(
        model_size=ModelSize.SMALL_3B,
        model_family=ModelFamily.QWEN,
    )

    model = loaded.model
    processor = loaded.processor
    adapter = loaded.adapter

    print("Model loaded.")

    # Load the same dataset used by the tutorial.
    examples = load_color_examples()

    SHOWCASE_OBJECT = "Banana"
    example = next(
        (e for e in examples if e.object == SHOWCASE_OBJECT),
        examples[0],
    )

    original_color = example.correct_answer[0]
    counterfact_color = example.incorrect_answer

    print()
    print("Object:              ", example.object)
    print("Original color:      ", original_color)
    print("Counterfactual color:", counterfact_color)

    # Same prior prompt used by the tutorial.
    question_prior = VMIRunnerBase._get_question(
        example,
        GroundingMode.PRIOR,
    )

    print("Prior prompt:", question_prior)

    # Use the counterfactual image.
    image = example.counterfact_image.pil_image

    inputs = adapter.build_inputs(
        processor=processor,
        image=image,
        question=question_prior,
        device=model.device,
    )

    # Get the token IDs for the two candidate colors.
    original_ids = processor.tokenizer(
        original_color,
        add_special_tokens=False,
    ).input_ids

    counterfact_ids = processor.tokenizer(
        counterfact_color,
        add_special_tokens=False,
    ).input_ids

    id_original = original_ids[0]
    id_counterfact = counterfact_ids[0]
    print("Original token IDs:", original_ids)
    print("Counterfactual token IDs:", counterfact_ids)

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline = get_logits(model, inputs)

    print()
    print("=" * 60)
    print("BASELINE")
    print("=" * 60)

    print(
        f"{original_color}: "
        f"{baseline[id_original].item():.4f}"
    )

    print(
        f"{counterfact_color}: "
        f"{baseline[id_counterfact].item():.4f}"
    )

    # ---------------------------------------------------------
    # Head-by-head weighting experiment
    # ---------------------------------------------------------

    weights = [1.0, 0.75, 0.50, 0.25, 0.0]

    print()
    print("=" * 70)
    print("HEAD-BY-HEAD WEIGHTING")
    print("=" * 70)

    for head in PROMOTING_HEADS:

        layer, head_idx = head

        print()
        print("-" * 70)
        print(f"L{layer}_H{head_idx}")
        print("-" * 70)

        for weight in weights:

            if weight == 1.0:
                logits = baseline

            else:
                with head_weighting(
                    model,
                    adapter,
                    [head],
                    weight,
                ):
                    logits = get_logits(model, inputs)

            top_id = logits.argmax().item()

            top_token = processor.tokenizer.decode(
                [top_id]
            ).strip()

            prior_logit = logits[id_original].item()
            visual_logit = logits[id_counterfact].item()

            delta = visual_logit - prior_logit

            print(
                f"weight={weight:.2f} | "
                f"top={top_token!r:15s} | "
                f"prior={prior_logit:9.4f} | "
                f"visual={visual_logit:9.4f} | "
                f"visual-prior={delta:+9.4f}"
            )