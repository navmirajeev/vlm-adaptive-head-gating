import torch
import json
import os

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

OUTPUT_FILE = "head_screening_results.json"


def collect_head_vectors(model, adapter, inputs):

    captured = {}
    handles = []

    layers = adapter.get_language_layer_modules(model)
    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    def make_hook(layer_idx):

        def hook(module, args):

            hidden = args[0]

            if hidden.dim() == 3:
                last_token = hidden[0, -1, :]
            else:
                last_token = hidden[-1, :]

            for head_idx in range(num_heads):

                if (layer_idx, head_idx) not in PROMOTING_HEADS:
                    continue

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
            o_proj.register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )

    try:

        with torch.no_grad():
            outputs = model(**inputs)

    finally:

        for handle in handles:
            handle.remove()

    return captured, outputs


def load_previous_results():

    if not os.path.exists(OUTPUT_FILE):
        return []

    try:

        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)

        print(
            f"Found previous results: "
            f"{len(results)} completed examples."
        )

        return results

    except Exception as e:

        print("Could not read previous results.")
        print("Starting from scratch.")
        print(e)

        return []


def save_results(results):

    # Write to a temporary file first.
    # This prevents losing the main file if something
    # goes wrong during saving.

    temp_file = OUTPUT_FILE + ".tmp"

    with open(temp_file, "w") as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    os.replace(temp_file, OUTPUT_FILE)

    print(
        f"  ✓ Saved {len(results)} examples "
        f"to {OUTPUT_FILE}"
    )


def main():

    print("=" * 70)
    print("FAST HEAD SCREENING")
    print("=" * 70)

    # --------------------------------------------------
    # Load previous results
    # --------------------------------------------------

    results = load_previous_results()

    completed_objects = {
        result["object"]
        for result in results
    }

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Load examples
    # --------------------------------------------------

    examples = load_color_examples()[:20]

    print()
    print("Total examples:", len(examples))
    print("Already completed:", len(completed_objects))
    print("Remaining:", len(examples) - len(completed_objects))

    # --------------------------------------------------
    # Process examples
    # --------------------------------------------------

    for example_idx, example in enumerate(
        examples,
        start=1,
    ):

        # Skip examples already saved

        if example.object in completed_objects:

            print()
            print(
                f"[{example_idx}/{len(examples)}] "
                f"{example.object} - SKIPPING (already saved)"
            )

            continue

        print()
        print("=" * 70)
        print(
            f"[{example_idx}/{len(examples)}] "
            f"{example.object}"
        )
        print("=" * 70)

        image = example.counterfact_image.pil_image

        question_prior = VMIRunnerBase._get_question(
            example,
            GroundingMode.PRIOR,
        )

        question_visual = VMIRunnerBase._get_question(
            example,
            GroundingMode.VISUAL,
        )

        # --------------------------------------------------
        # Build inputs
        # --------------------------------------------------

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

        # --------------------------------------------------
        # PRIOR FORWARD
        # --------------------------------------------------

        print("  Running prior condition...")

        prior_heads, _ = collect_head_vectors(
            model,
            adapter,
            inputs_prior,
        )

        # --------------------------------------------------
        # VISUAL FORWARD
        # --------------------------------------------------

        print("  Running visual condition...")

        visual_heads, _ = collect_head_vectors(
            model,
            adapter,
            inputs_visual,
        )

        # --------------------------------------------------
        # Compare heads
        # --------------------------------------------------

        example_result = {
            "object": example.object,
            "prior_color": example.correct_answer[0],
            "visual_color": example.incorrect_answer,
            "heads": {},
        }

        for layer_idx, head_idx in PROMOTING_HEADS:

            key = (layer_idx, head_idx)

            prior = prior_heads[key]
            visual = visual_heads[key]

            difference = visual - prior

            difference_norm = difference.norm().item()

            cosine = torch.nn.functional.cosine_similarity(
                prior.unsqueeze(0),
                visual.unsqueeze(0),
            ).item()

            example_result["heads"][
                f"L{layer_idx}_H{head_idx}"
            ] = {
                "prior_norm": prior.norm().item(),
                "visual_norm": visual.norm().item(),
                "difference_norm": difference_norm,
                "cosine": cosine,
            }

            print(
                f"  L{layer_idx} H{head_idx:02d} | "
                f"diff={difference_norm:8.4f} | "
                f"cos={cosine:7.4f}"
            )

        # --------------------------------------------------
        # ADD RESULT
        # --------------------------------------------------

        results.append(example_result)
        completed_objects.add(example.object)

        # --------------------------------------------------
        # SAVE IMMEDIATELY
        # --------------------------------------------------

        save_results(results)

        print(
            f"  Finished {example.object}."
        )

    # --------------------------------------------------
    # DONE
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    print(
        f"Saved {len(results)} examples "
        f"to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()