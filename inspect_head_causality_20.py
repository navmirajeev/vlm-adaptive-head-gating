import torch
import statistics

from vdpo.types.enums import ModelFamily, ModelSize, GroundingMode
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


NUM_EXAMPLES = 20


def get_logits(model, inputs, prior_token, visual_token):

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1]

    prior_logit = logits[prior_token].item()
    visual_logit = logits[visual_token].item()

    return prior_logit, visual_logit


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

    examples = load_color_examples()

    # Use first 20 examples
    examples = examples[:NUM_EXAMPLES]

    tokenizer = processor.tokenizer

    layers = adapter.get_language_layer_modules(model)

    results = {
        head: []
        for head in PROMOTING_HEADS
    }

    correct_count = 0
    wrong_count = 0

    # =========================================================
    # Process examples
    # =========================================================

    for example_idx, example in enumerate(examples, 1):

        print()
        print("=" * 70)
        print(
            f"[{example_idx}/{len(examples)}] "
            f"{example.object}"
        )
        print("=" * 70)

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

        prior_answer = example.correct_answer[0]
        visual_answer = example.incorrect_answer

        # -----------------------------------------------------
        # Tokens
        # -----------------------------------------------------

        prior_tokens = tokenizer.encode(
            prior_answer,
            add_special_tokens=False,
        )

        visual_tokens = tokenizer.encode(
            visual_answer,
            add_special_tokens=False,
        )

        if not prior_tokens or not visual_tokens:
            print("Skipping: could not tokenize answer.")
            continue

        prior_token = prior_tokens[0]
        visual_token = visual_tokens[0]

        # -----------------------------------------------------
        # Baseline
        # -----------------------------------------------------

        baseline_prior, baseline_visual = get_logits(
            model,
            inputs,
            prior_token,
            visual_token,
        )

        baseline_difference = (
            baseline_visual - baseline_prior
        )

        predicted_token = (
            visual_token
            if baseline_visual > baseline_prior
            else prior_token
        )

        predicted_text = tokenizer.decode(
            [predicted_token]
        ).strip()

        is_correct = (
            predicted_token == visual_token
        )

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        print("Prior:", prior_answer)
        print("Visual:", visual_answer)
        print("Predicted:", predicted_text)

        print(
            f"Baseline visual-prior difference: "
            f"{baseline_difference:.4f}"
        )

        # -----------------------------------------------------
        # Test every head
        # -----------------------------------------------------

        for layer_idx, head_idx in PROMOTING_HEADS:

            attn = layers[layer_idx].self_attn

            original_forward = attn.forward

            def make_modified_forward(
                original_forward,
                attn,
                head_idx,
            ):

                def modified_forward(*args, **kwargs):

                    output = original_forward(
                        *args,
                        **kwargs,
                    )

                    hidden_states = output[0]

                    batch, seq, hidden = (
                        hidden_states.shape
                    )

                    head_dim = (
                        hidden // attn.num_heads
                    )

                    start = head_idx * head_dim
                    end = start + head_dim

                    hidden_states = (
                        hidden_states.clone()
                    )

                    hidden_states[
                        :, :, start:end
                    ] = 0

                    return (
                        hidden_states,
                        *output[1:],
                    )

                return modified_forward

            attn.forward = make_modified_forward(
                original_forward,
                attn,
                head_idx,
            )

            try:

                intervention_prior, intervention_visual = (
                    get_logits(
                        model,
                        inputs,
                        prior_token,
                        visual_token,
                    )
                )

            finally:

                attn.forward = original_forward

            intervention_difference = (
                intervention_visual
                - intervention_prior
            )

            delta = (
                intervention_difference
                - baseline_difference
            )

            results[
                (layer_idx, head_idx)
            ].append(delta)

        # -----------------------------------------------------
        # Print progress
        # -----------------------------------------------------

        print("Head effects:")

        for head in PROMOTING_HEADS:

            delta = results[head][-1]

            print(
                f"  L{head[0]} H{head[1]:02d}: "
                f"{delta:+.4f}"
            )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("=" * 70)
    print("CAUSAL HEAD SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Examples processed: "
        f"{correct_count + wrong_count}"
    )

    print(
        f"Visual preferred at baseline: "
        f"{correct_count}"
    )

    print(
        f"Prior preferred at baseline: "
        f"{wrong_count}"
    )

    print()

    print(
        f"{'Head':<12}"
        f"{'Mean Δ':>12}"
        f"{'Median Δ':>12}"
        f"{'Positive':>12}"
        f"{'Negative':>12}"
    )

    print("-" * 60)

    for head in PROMOTING_HEADS:

        values = results[head]

        if not values:
            continue

        mean_delta = statistics.mean(values)

        median_delta = statistics.median(values)

        positive = sum(
            x > 0 for x in values
        )

        negative = sum(
            x < 0 for x in values
        )

        print(
            f"L{head[0]} H{head[1]:02d}"
            f"{mean_delta:>12.4f}"
            f"{median_delta:>12.4f}"
            f"{positive:>12}"
            f"{negative:>12}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()