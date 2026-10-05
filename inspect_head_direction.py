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


def get_token_logit(logits, tokenizer, word):
    """
    Get the logit for the first token corresponding to a color word.
    """

    variants = {
        word,
        word.lower(),
        word.capitalize(),
    }

    token_ids = []

    for variant in variants:
        encoded = tokenizer(
            variant,
            add_special_tokens=False,
        )

        if len(encoded.input_ids) > 0:
            token_ids.append(encoded.input_ids[0])

    if not token_ids:
        return None

    return max(
        logits[token_id].item()
        for token_id in token_ids
    )


def get_color_logits(model, processor, inputs, colors):

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    result = {}

    for color in colors:
        result[color] = get_token_logit(
            logits,
            processor.tokenizer,
            color,
        )

    return result


def collect_head_vectors(model, adapter, inputs):

    layers = adapter.get_language_layer_modules(model)

    captured = {}
    handles = []

    def make_hook(layer_idx, head_idx):

        def hook(module, args):

            hidden = args[0]

            if hidden.dim() == 3:
                last_token = hidden[0, -1, :]
            else:
                last_token = hidden[-1, :]

            head_dim = 128

            start = head_idx * head_dim
            end = start + head_dim

            head_vector = last_token[start:end]

            captured[(layer_idx, head_idx)] = (
                head_vector.detach()
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

    # Small diagnostic run first.
    examples = examples[:20]

    print()
    print("=" * 70)
    print("HEAD DIRECTION DIAGNOSTIC")
    print("=" * 70)

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

        prior_color = example.correct_answer[0]
        visual_color = example.incorrect_answer

        colors = [
            prior_color,
            visual_color,
        ]

        logits = get_color_logits(
            model,
            processor,
            inputs,
            colors,
        )

        prior_logit = logits[prior_color]
        visual_logit = logits[visual_color]

        baseline_direction = (
            visual_logit - prior_logit
        )

        predicted = max(
            logits,
            key=logits.get,
        )

        expected = visual_color

        correct = (
            predicted.lower()
            == expected.lower()
        )

        print(
            f"Prior color:  {prior_color}"
        )

        print(
            f"Visual color: {visual_color}"
        )

        print(
            f"Prediction:   {predicted}"
        )

        print(
            f"Status:       "
            f"{'CORRECT' if correct else 'WRONG'}"
        )

        print(
            f"Prior logit:  {prior_logit:.4f}"
        )

        print(
            f"Visual logit: {visual_logit:.4f}"
        )

        print(
            f"Visual-prior logit difference: "
            f"{baseline_direction:.4f}"
        )

        # --------------------------------------------------
        # Capture head vectors
        # --------------------------------------------------

        heads = collect_head_vectors(
            model,
            adapter,
            inputs,
        )

        print()
        print("Head activation norms:")

        for layer_idx, head_idx in PROMOTING_HEADS:

            vector = heads[
                (layer_idx, head_idx)
            ]

            norm = vector.norm().item()

            print(
                f"  L{layer_idx:02d} H{head_idx:02d}: "
                f"{norm:.4f}"
            )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()