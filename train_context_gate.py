import json
import os

import torch

from vdpo.representation_gates import HeadContextGate
from vdpo.representation_intervention import adaptive_head_context_gate
from vdpo.types.enums import GroundingMode, ModelFamily, ModelSize
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
EPOCHS = 1
LEARNING_RATE = 1e-3

CHECKPOINT_JSON = "training_context_gate_checkpoint.json"
GATES_FILE = "trained_head_context_gates.pt"


def get_logits(model, inputs):
    outputs = model(**inputs)
    return outputs.logits[0, -1, :]


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


def save_checkpoint(
    epoch,
    completed_examples,
    loss_records,
    gates,
    context_projector,
    optimizer,
):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            {
                "epoch": epoch,
                "completed_examples": completed_examples,
                "loss_records": loss_records,
            },
            file,
            indent=2,
        )

    gate_state = {
        f"L{layer}_H{head}": gates[(layer, head)].state_dict()
        for layer, head in PROMOTING_HEADS
    }

    torch.save(
        {
            "gates": gate_state,
            "context_projector": context_projector.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        GATES_FILE,
    )


def main():
    print("=" * 70)
    print("HEAD + SAME-TOKEN CONTEXT GATE TRAINING")
    print("=" * 70)

    print("\nLoading Qwen2.5-VL 3B...")

    loaded = load_vlm(
        model_size=ModelSize.SMALL_3B,
        model_family=ModelFamily.QWEN,
    )

    model = loaded.model
    processor = loaded.processor
    adapter = loaded.adapter

    print("Model loaded.")

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print("Qwen parameters frozen.")

    # Keep the trainable gate modules in FP32.
    gate_device = model.device

    context_projector = torch.nn.Sequential(
        torch.nn.Linear(2048, 32),
        torch.nn.GELU(),
    ).to(device=gate_device, dtype=torch.float32)

    gates = {
        head: HeadContextGate(
            head_dim=128,
            context_dim=32,
            hidden_dim=8,
        ).to(device=gate_device, dtype=torch.float32)
        for head in PROMOTING_HEADS
    }

    print(f"\nCreated {len(gates)} contextual head gates.")
    print("Context projector: 2048 -> 32")

    trainable_parameters = [
        parameter
        for gate in gates.values()
        for parameter in gate.parameters()
    ]
    trainable_parameters.extend(context_projector.parameters())

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable_parameters),
    )

    examples = load_color_examples()[:NUM_EXAMPLES]
    print(f"\nTraining examples: {len(examples)}")

    start_epoch = 0
    completed_examples = 0
    loss_records = []

    # Load saved parameters and progress, if available.
    if os.path.exists(CHECKPOINT_JSON) and os.path.exists(GATES_FILE):
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        saved = torch.load(
            GATES_FILE,
            map_location=gate_device,
            weights_only=False,
        )

        for head in PROMOTING_HEADS:
            key = f"L{head[0]}_H{head[1]}"
            gates[head].load_state_dict(saved["gates"][key])

        context_projector.load_state_dict(saved["context_projector"])
        optimizer.load_state_dict(saved["optimizer"])

        start_epoch = checkpoint.get("epoch", 0)
        completed_examples = checkpoint.get("completed_examples", 0)
        loss_records = checkpoint.get("loss_records", [])

        print("\nCheckpoint restored.")
        print(f"Saved epoch: {start_epoch}")
        print(f"Completed examples: {completed_examples}/{len(examples)}")

    # If the single planned epoch has already finished, only report results.
    training_complete = (
        start_epoch >= EPOCHS
        or completed_examples >= len(examples)
    )

    if training_complete:
        print("\nTraining already completed. Skipping training.")
    else:
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        for epoch in range(start_epoch, EPOCHS):
            first_example = completed_examples if epoch == start_epoch else 0

            for example_idx in range(first_example, len(examples)):
                example = examples[example_idx]

                print(
                    f"\nEpoch {epoch + 1}/{EPOCHS} | "
                    f"Example {example_idx + 1}/{len(examples)} | "
                    f"{example.object}"
                )

                inputs = get_example_inputs(
                    example, model, processor, adapter
                )

                prior_ids = processor.tokenizer(
                    example.correct_answer[0],
                    add_special_tokens=False,
                ).input_ids

                visual_ids = processor.tokenizer(
                    example.incorrect_answer,
                    add_special_tokens=False,
                ).input_ids

                if not prior_ids or not visual_ids:
                    print("Skipping example: answer tokenization was empty.")
                    continue

                prior_id = prior_ids[0]
                visual_id = visual_ids[0]

                optimizer.zero_grad(set_to_none=True)

                with adaptive_head_context_gate(
                    model=model,
                    adapter=adapter,
                    heads=PROMOTING_HEADS,
                    gates=gates,
                    context_projector=context_projector,
                ):
                    logits = get_logits(model, inputs)

                visual_logit = logits[visual_id]
                prior_logit = logits[prior_id]
                margin = visual_logit - prior_logit

                loss = torch.nn.functional.softplus(-margin)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                margin_value = margin.detach().item()

                loss_records.append({
                    "epoch": epoch + 1,
                    "example": example_idx + 1,
                    "object": example.object,
                    "loss": loss_value,
                    "margin": margin_value,
                })

                print(f"Loss: {loss_value:.6f}")
                print(f"Visual - Prior: {margin_value:.6f}")

                # Save after each processed example.
                save_checkpoint(
                    epoch=epoch + 1,
                    completed_examples=example_idx + 1,
                    loss_records=loss_records,
                    gates=gates,
                    context_projector=context_projector,
                    optimizer=optimizer,
                )
                print("Checkpoint saved.")

            completed_examples = 0
            start_epoch = epoch + 1

    # Final diagnostic output.
    print("\n" + "=" * 70)
    print("FINAL GATE OUTPUTS (ZERO INPUTS)")
    print("=" * 70)

    for head in PROMOTING_HEADS:
        gate = gates[head]
        parameter = next(gate.parameters())

        dummy_head = torch.zeros(
            1, 128,
            device=parameter.device,
            dtype=parameter.dtype,
        )

        dummy_context = torch.zeros(
            1, 32,
            device=parameter.device,
            dtype=parameter.dtype,
        )

        gate.eval()
        with torch.no_grad():
            weight = gate(dummy_head, dummy_context)[0, 0].item()

        print(f"L{head[0]}_H{head[1]}: {weight:.4f}")

    # Final save. Preserve the existing trained weights.
    save_checkpoint(
        epoch=EPOCHS,
        completed_examples=len(examples),
        loss_records=loss_records,
        gates=gates,
        context_projector=context_projector,
        optimizer=optimizer,
    )

    print("\nCHECKPOINT SAVED")
    print(f"Progress: {len(examples)}/{len(examples)}")
    print(f"JSON: {CHECKPOINT_JSON}")
    print(f"Gates: {GATES_FILE}")
    print("\n" + "=" * 70)
    print("CONTEXT GATE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

