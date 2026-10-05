import torch

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

    # Start small so we can verify the experiment works.
    examples = examples[:20]

    layers = adapter.get_language_layer_modules(model)

    print()
    print("=" * 70)
    print("TESTING HEAD BEHAVIOR")
    print("=" * 70)
    print(f"Number of examples: {len(examples)}")

    def get_prediction(inputs):
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]
        token_id = logits.argmax().item()

        token = processor.tokenizer.decode(
            [token_id]
        ).strip()

        return token

    def collect_heads(inputs):

        captured = {}
        handles = []

        def make_hook(layer_idx, head_idx):
            def hook(module, args):

                hidden = args[0]

                if hidden.dim() == 3:
                    last_token = hidden[0, -1, :]
                else:
                    last_token = hidden[-1, :]

                num_heads = 16
                head_dim = 128

                start = head_idx * head_dim
                end = start + head_dim

                head_output = last_token[start:end]

                captured[(layer_idx, head_idx)] = (
                    head_output.detach()
                    .float()
                    .cpu()
                )

            return hook
        
        for layer_idx, head_idx in PROMOTING_HEADS:

            o_proj = layers[layer_idx].self_attn.o_proj

            handles.append(
                o_proj.register_forward_pre_hook(
                    make_hook(layer_idx, head_idx)
                )
            )

        try:
            with torch.no_grad():
                model(**inputs)

        finally:
            for handle in handles:
                handle.remove()

        return captured

    results = []

    for i, example in enumerate(examples):

        print()
        print(
            f"[{i + 1}/{len(examples)}] "
            f"{example.object}"
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

        predicted = get_prediction(inputs)

        expected_answer = example.incorrect_answer.strip().lower()

        is_correct = (
            predicted.lower() == expected_answer
        )

        status = (
            "CORRECT"
            if is_correct
            else "WRONG"
        )

        print(
            f"Expected:  {example.incorrect_answer}"
        )
        print(
            f"Predicted: {predicted}"
        )
        print(
            f"Status:    {status}"
        )

        heads = collect_heads(inputs)

        head_norms = {}

        for layer_idx, head_idx in PROMOTING_HEADS:

            vector = heads[(layer_idx, head_idx)]

            head_norms[(layer_idx, head_idx)] = (
                vector.norm().item()
            )

        results.append(
            {
                "object": example.object,
                "expected": example.incorrect_answer,
                "predicted": predicted,
                "correct": is_correct,
                "head_norms": head_norms,
            }
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    correct_results = [
        r for r in results
        if r["correct"]
    ]

    wrong_results = [
        r for r in results
        if not r["correct"]
    ]

    print(
        f"Correct: {len(correct_results)}"
    )

    print(
        f"Wrong:   {len(wrong_results)}"
    )

    print()
    print("=" * 70)
    print("HEAD ACTIVATION COMPARISON")
    print("=" * 70)

    for layer_idx, head_idx in PROMOTING_HEADS:

        correct_values = [
            r["head_norms"][(layer_idx, head_idx)]
            for r in correct_results
        ]

        wrong_values = [
            r["head_norms"][(layer_idx, head_idx)]
            for r in wrong_results
        ]

        if correct_values:
            correct_mean = sum(correct_values) / len(
                correct_values
            )
        else:
            correct_mean = float("nan")

        if wrong_values:
            wrong_mean = sum(wrong_values) / len(
                wrong_values
            )
        else:
            wrong_mean = float("nan")

        print(
            f"Layer {layer_idx:2d}, "
            f"Head {head_idx:2d} | "
            f"Correct mean = {correct_mean:8.4f} | "
            f"Wrong mean = {wrong_mean:8.4f}"
        )


if __name__ == "__main__":
    main()