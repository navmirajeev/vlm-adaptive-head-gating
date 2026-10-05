import os
import json
import torch

from vdpo.adaptive_gate import AdaptiveHeadGate
from vdpo.gate_features import (
    load_gate_features,
    PROMOTING_HEADS,
)

from vdpo.interventions import (
    adaptive_head_weighting_trainable,
)

from vdpo.types.enums import (
    GroundingMode,
    ModelFamily,
    ModelSize,
)

from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


CHECKPOINT_FILE = "adaptive_gate_checkpoint.pt"

LEARNING_RATE = 1e-3

NUM_EPOCHS = 3

START_EXAMPLE = 0


def get_token_logit(
    model,
    inputs,
    token_id,
):
    """
    Run Qwen and return the logit for one token.
    """

    outputs = model(**inputs)

    return outputs.logits[0, -1, token_id]


def main():

    print("=" * 70)
    print("ADAPTIVE GATE - STAGE A TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load features
    # ---------------------------------------------------------

    features, metadata = load_gate_features(
        "head_screening_results.json"
    )

    print()
    print("Feature shape:", tuple(features.shape))

    # ---------------------------------------------------------
    # Load examples
    # ---------------------------------------------------------

    examples = load_color_examples()[:20]

    print(
        "Examples:",
        len(examples),
    )

    # ---------------------------------------------------------
    # Create gate
    # ---------------------------------------------------------

    gate = AdaptiveHeadGate(
        input_dim=features.shape[1],
        num_heads=len(PROMOTING_HEADS),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # Checkpoint loading
    # ---------------------------------------------------------

    start_epoch = 0
    start_example = START_EXAMPLE

    if os.path.exists(CHECKPOINT_FILE):

        print()
        print(
            "Checkpoint found:",
            CHECKPOINT_FILE,
        )

        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location="cpu",
        )

        gate.load_state_dict(
            checkpoint["gate_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        start_epoch = checkpoint["epoch"]

        start_example = checkpoint[
            "example_idx"
        ]

        print(
            "Resuming from epoch:",
            start_epoch,
        )

        print(
            "Resuming from example:",
            start_example + 1,
        )

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

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    print()
    print("Qwen parameters frozen.")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
        )
        print("=" * 70)

        epoch_loss = 0.0

        epoch_examples = 0

        # Resume only applies to the first
        # epoch being resumed.

        example_start = (
            start_example
            if epoch == start_epoch
            else 0
        )

        for idx in range(
            example_start,
            len(examples),
        ):

            example = examples[idx]

            feature_vector = features[idx]

            print()
            print(
                f"Example {idx + 1}/"
                f"{len(examples)}: "
                f"{example.object}"
            )

            # -------------------------------------------------
            # Generate gate weights
            # -------------------------------------------------

            weights = gate(
                feature_vector.unsqueeze(0)
            )[0]

            # -------------------------------------------------
            # Colors
            # -------------------------------------------------

            prior_color = (
                example.correct_answer[0]
            )

            visual_color = (
                example.incorrect_answer
            )

            # -------------------------------------------------
            # Build question
            # -------------------------------------------------

            question = VMIRunnerBase._get_question(
                example,
                GroundingMode.PRIOR,
            )

            image = (
                example.counterfact_image.pil_image
            )

            inputs = adapter.build_inputs(
                processor=processor,
                image=image,
                question=question,
                device=model.device,
            )

            # -------------------------------------------------
            # Token IDs
            # -------------------------------------------------

            prior_id = processor.tokenizer(
                prior_color,
                add_special_tokens=False,
            ).input_ids[0]

            visual_id = processor.tokenizer(
                visual_color,
                add_special_tokens=False,
            ).input_ids[0]

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            optimizer.zero_grad()

            with adaptive_head_weighting_trainable(
                model,
                adapter,
                PROMOTING_HEADS,
                weights,
            ):

                visual_logit = get_token_logit(
                    model,
                    inputs,
                    visual_id,
                )

                prior_logit = get_token_logit(
                    model,
                    inputs,
                    prior_id,
                )

                # -------------------------------------------------
                # Stage-A objective
                #
                # Encourage visual answer to have a larger
                # logit than the prior answer.
                # -------------------------------------------------

                margin = (
                    visual_logit -
                    prior_logit
                )

                loss = -margin

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            loss.backward()

            optimizer.step()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            loss_value = loss.item()

            margin_value = margin.item()

            mean_weight = (
                weights.detach().mean().item()
            )

            epoch_loss += loss_value

            epoch_examples += 1

            print(
                f"Prior logit: "
                f"{prior_logit.item():.4f}"
            )

            print(
                f"Visual logit: "
                f"{visual_logit.item():.4f}"
            )

            print(
                f"Margin: "
                f"{margin_value:.4f}"
            )

            print(
                f"Loss: "
                f"{loss_value:.4f}"
            )

            print(
                f"Mean gate weight: "
                f"{mean_weight:.4f}"
            )

            # -------------------------------------------------
            # Save checkpoint AFTER every example
            # -------------------------------------------------

            torch.save(
                {
                    "epoch": epoch,
                    "example_idx": idx + 1,
                    "gate_state": gate.state_dict(),
                    "optimizer_state": (
                        optimizer.state_dict()
                    ),
                    "loss": loss_value,
                    "margin": margin_value,
                },
                CHECKPOINT_FILE,
            )

            print(
                "Checkpoint saved."
            )

        # -----------------------------------------------------
        # Epoch summary
        # -----------------------------------------------------

        average_loss = (
            epoch_loss /
            max(epoch_examples, 1)
        )

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Average loss: "
            f"{average_loss:.4f}"
        )

        print("-" * 70)

        # Reset resume position after first
        # resumed epoch.

        start_example = 0

    # ---------------------------------------------------------
    # Save final gate
    # ---------------------------------------------------------

    torch.save(
        gate.state_dict(),
        "adaptive_gate_trained.pt",
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Saved:",
        "adaptive_gate_trained.pt",
    )


if __name__ == "__main__":
    main()