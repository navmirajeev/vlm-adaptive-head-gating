import torch

from vdpo.types.enums import ModelFamily, ModelSize, GroundingMode
from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase
from nnsight import NNsight


# Prior-promoting heads identified by the original paper
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

    nnsight = NNsight(model)

    print("Model loaded.")

    # ---------------------------------------------------------
    # Load Banana example
    # ---------------------------------------------------------

    examples = load_color_examples()

    example = next(
        e for e in examples
        if e.object == "Banana"
    )

    image = example.counterfact_image.pil_image

    question_prior = VMIRunnerBase._get_question(
        example,
        GroundingMode.PRIOR,
    )

    question_visual = VMIRunnerBase._get_question(
        example,
        GroundingMode.VISUAL,
    )

    inputs_prior = adapter.build_inputs(
        processor=processor,
        image=image,
        question=question_prior,
        device=model.device,
    )

    inputs_visual = adapter.build_inputs(
        processor=processor,
        image=image,
        question=question_visual,
        device=model.device,
    )

    print()
    print("Object:", example.object)
    print("Original color:", example.correct_answer[0])
    print("Counterfactual color:", example.incorrect_answer)

    print()
    print("Prior question:")
    print(question_prior)

    print()
    print("Visual question:")
    print(question_visual)

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads

    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    print()
    print("Number of attention heads:", num_heads)
    print("Head dimension:", head_dim)

    layers = adapter.get_language_layer_modules(model)

    # ---------------------------------------------------------
    # Collect head representations
    # ---------------------------------------------------------

    def collect(inputs, label):
        captured = {}

        handles = []

        layers = adapter.get_language_layer_modules(model)

        def make_hook(layer_idx):
            def hook(module, args):
                hidden = args[0]

                if hidden.dim() == 3:
                    last_token = hidden[0, -1, :]
                else:
                    last_token = hidden[-1, :]

                for head_idx in range(num_heads):
                    start = head_idx * head_dim
                    end = start + head_dim

                    captured[(layer_idx, head_idx)] = (
                        last_token[start:end]
                        .detach()
                        .float()
                        .cpu()
                    )

            return hook

        for layer_idx in range(len(layers)):
            o_proj = layers[layer_idx].self_attn.o_proj

            handles.append(
                o_proj.register_forward_pre_hook(make_hook(layer_idx))
            )

        try:
            with torch.no_grad():
                model(**inputs)

        finally:
            for handle in handles:
                handle.remove()

        return captured

    # ---------------------------------------------------------
    # Prior prompt
    # ---------------------------------------------------------

    prior_heads = collect(
        inputs_prior,
        "PRIOR PROMPT",
    )

    # ---------------------------------------------------------
    # Visual prompt
    # ---------------------------------------------------------

    visual_heads = collect(
        inputs_visual,
        "VISUAL PROMPT",
    )

    # ---------------------------------------------------------
    # Compare representations
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PRIOR VS VISUAL DIFFERENCE")
    print("=" * 70)

    for key in PROMOTING_HEADS:

        prior = prior_heads[key].detach().float()
        visual = visual_heads[key].detach().float()

        difference = visual - prior

        cosine = torch.nn.functional.cosine_similarity(
            prior.unsqueeze(0),
            visual.unsqueeze(0),
        ).item()

        print(
            f"Layer {key[0]:2d}, Head {key[1]:2d} | "
            f"difference norm={difference.norm(): .6f} | "
            f"cosine={cosine: .6f}"
        )


if __name__ == "__main__":
    main()