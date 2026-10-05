import torch

from vdpo.types.enums import ModelFamily, ModelSize, GroundingMode
from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


# Heads identified by the original causal analysis
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

    # ---------------------------------------------------------
    # Load ONE example
    # ---------------------------------------------------------

    examples = load_color_examples()

    example = next(
        e for e in examples
        if e.object == "Vulture"
    )

    image = example.counterfact_image.pil_image

    question = VMIRunnerBase._get_question(
        example,
        GroundingMode.VISUAL,
    )

    inputs = adapter.build_inputs(
        processor=processor,
        image=image,
        question=question,
        device=model.device,
    )

    visual_answer = example.incorrect_answer
    prior_answer = example.correct_answer[0]

    print()
    print("=" * 70)
    print("CAUSAL HEAD TEST")
    print("=" * 70)

    print("Object:", example.object)
    print("Prior answer:", prior_answer)
    print("Visual answer:", visual_answer)

    # ---------------------------------------------------------
    # Get token IDs
    # ---------------------------------------------------------

    tokenizer = processor.tokenizer

    prior_tokens = tokenizer.encode(
        prior_answer,
        add_special_tokens=False,
    )

    visual_tokens = tokenizer.encode(
        visual_answer,
        add_special_tokens=False,
    )

    prior_token = prior_tokens[0]
    visual_token = visual_tokens[0]

    print()
    print("Prior token:", prior_token)
    print("Visual token:", visual_token)

    # ---------------------------------------------------------
    # Get model output
    # ---------------------------------------------------------

    def run_model():

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1]

        prior_logit = logits[prior_token].item()
        visual_logit = logits[visual_token].item()

        return prior_logit, visual_logit

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_prior, baseline_visual = run_model()

    baseline_difference = baseline_visual - baseline_prior

    print()
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)

    print(f"Prior logit:  {baseline_prior:.4f}")
    print(f"Visual logit: {baseline_visual:.4f}")
    print(f"Visual-prior difference: {baseline_difference:.4f}")

    # ---------------------------------------------------------
    # Find language layers
    # ---------------------------------------------------------

    layers = adapter.get_language_layer_modules(model)

    # ---------------------------------------------------------
    # Test each head
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("HEAD INTERVENTIONS")
    print("=" * 70)

    for layer_idx, head_idx in PROMOTING_HEADS:

        print()
        print(
            f"Testing Layer {layer_idx}, Head {head_idx}..."
        )

        attn = layers[layer_idx].self_attn

        original_forward = attn.forward

        def modified_forward(*args, **kwargs):

            output = original_forward(*args, **kwargs)

            # Qwen attention normally returns:
            # (hidden_states, attention_weights, ...)
            hidden_states = output[0]

            batch, seq, hidden = hidden_states.shape

            head_dim = hidden // attn.num_heads

            start = head_idx * head_dim
            end = start + head_dim

            hidden_states = hidden_states.clone()

            hidden_states[:, :, start:end] = 0

            return (
                hidden_states,
                *output[1:],
            )

        attn.forward = modified_forward

        try:

            intervention_prior, intervention_visual = run_model()

        finally:

            attn.forward = original_forward

        intervention_difference = (
            intervention_visual - intervention_prior
        )

        change = (
            intervention_difference
            - baseline_difference
        )

        print(
            f"Layer {layer_idx}, Head {head_idx}"
        )

        print(
            f"  Prior logit:       {intervention_prior:.4f}"
        )

        print(
            f"  Visual logit:      {intervention_visual:.4f}"
        )

        print(
            f"  Visual-prior diff: {intervention_difference:.4f}"
        )

        print(
            f"  Change from base:  {change:+.4f}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()