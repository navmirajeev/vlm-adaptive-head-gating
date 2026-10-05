import json
import os

import torch
import torch.nn.functional as F

from vdpo.representation_gates import HeadOnlyGate
from vdpo.representation_intervention import adaptive_head_gate
from vdpo.types.enums import ModelFamily, ModelSize, GroundingMode
from vdpo.utils.load_model import load_vlm
from vdpo.utils.load_examples import load_color_examples
from vdpo.runner.core import VMIRunnerBase


# ============================================================
# CONFIGURATION
# ============================================================

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
RETENTION_WEIGHT = 0.05

CHECKPOINT_JSON = "training_checkpoint.json"
GATES_FILE = "trained_head_only_gates.pt"


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def save_checkpoint(
    gates,
    optimizer,
    epoch,
    completed_examples,
    loss_history,
):
    """
    Save training progress after every example.

    JSON:
        Human-readable checkpoint containing gate parameters,
        progress and loss history.

    PT:
        Native PyTorch checkpoint for easier restoration.
    """

    # --------------------------------------------------------
    # JSON representation of gate parameters
    # --------------------------------------------------------

    json_gates = {}

    for head, gate in gates.items():

        key = f"L{head[0]}_H{head[1]}"

        json_gates[key] = {
            name: parameter.detach().cpu().tolist()
            for name, parameter in gate.state_dict().items()
        }

    checkpoint = {
        "epoch": epoch,
        "completed_examples": completed_examples,
        "loss_history": loss_history,
        "gate_parameters": json_gates,
    }

    # --------------------------------------------------------
    # Atomic JSON save
    # --------------------------------------------------------

    temp_json = CHECKPOINT_JSON + ".tmp"

    with open(temp_json, "w") as f:

        json.dump(
            checkpoint,
            f,
            indent=2,
        )

    os.replace(
        temp_json,
        CHECKPOINT_JSON,
    )

    # --------------------------------------------------------
    # Native PyTorch save
    # --------------------------------------------------------

    torch.save(
        {
            "gates": {
                f"L{head[0]}_H{head[1]}":
                    gate.state_dict()
                for head, gate in gates.items()
            },
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "completed_examples": completed_examples,
            "loss_history": loss_history,
        },
        GATES_FILE,
    )

    print()
    print("  CHECKPOINT SAVED")
    print(
        f"  Progress: "
        f"{completed_examples}/{NUM_EXAMPLES}"
    )
    print(
        f"  JSON: {CHECKPOINT_JSON}"
    )
    print(
        f"  Gates: {GATES_FILE}"
    )


def load_checkpoint(
    gates,
    optimizer,
):
    """
    Restore the most recent checkpoint.

    Returns:

        epoch
        completed_examples
        loss_history
    """

    if not os.path.exists(CHECKPOINT_JSON):

        print()
        print("No previous checkpoint found.")
        print("Starting training from scratch.")

        return 0, 0, []

    print()
    print("=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    with open(
        CHECKPOINT_JSON,
        "r",
    ) as f:

        checkpoint = json.load(f)

    epoch = checkpoint.get(
        "epoch",
        0,
    )

    completed_examples = checkpoint.get(
        "completed_examples",
        0,
    )

    loss_history = checkpoint.get(
        "loss_history",
        [],
    )

    # --------------------------------------------------------
    # Restore gate parameters
    # --------------------------------------------------------

    gate_parameters = checkpoint.get(
        "gate_parameters",
        {},
    )

    for head, gate in gates.items():

        key = f"L{head[0]}_H{head[1]}"

        if key not in gate_parameters:
            continue

        restored_state = {}

        current_state = gate.state_dict()

        for name, values in gate_parameters[key].items():

            restored_state[name] = torch.tensor(
                values,
                dtype=current_state[name].dtype,
                device=current_state[name].device,
            )

        gate.load_state_dict(
            restored_state
        )

    # --------------------------------------------------------
    # Restore optimizer if PyTorch checkpoint exists
    # --------------------------------------------------------

    if os.path.exists(GATES_FILE):

        try:

            checkpoint_pt = torch.load(
                GATES_FILE,
                map_location="cpu",
                weights_only=False,
            )

            if "optimizer" in checkpoint_pt:

                optimizer.load_state_dict(
                    checkpoint_pt["optimizer"]
                )

                print(
                    "Optimizer state restored."
                )

        except Exception as e:

            print(
                "Warning: could not restore "
                f"optimizer state: {e}"
            )

    print(
        f"Previous epoch: {epoch}"
    )

    print(
        f"Previously completed examples: "
        f"{completed_examples}"
    )

    print(
        f"Previous loss records: "
        f"{len(loss_history)}"
    )

    # --------------------------------------------------------
    # If the previous epoch completed, start a new epoch
    # --------------------------------------------------------

    if completed_examples >= NUM_EXAMPLES:

        epoch += 1
        completed_examples = 0

        print()
        print(
            "Previous epoch was complete."
        )

        print(
            f"Starting epoch {epoch + 1}."
        )

    else:

        print()
        print(
            f"Resuming from example "
            f"{completed_examples + 1}."
        )

    return (
        epoch,
        completed_examples,
        loss_history,
    )


# ============================================================
# MODEL UTILITIES
# ============================================================

def get_logits(
    model,
    inputs,
):

    outputs = model(
        **inputs
    )

    return outputs.logits[
        0,
        -1,
        :
    ]


def get_token_id(
    processor,
    text,
):

    ids = processor.tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    if not ids:
        raise ValueError(
            f"Could not tokenize {text!r}"
        )

    return ids[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CHECKPOINT-SAFE HEAD-ONLY "
        "REPRESENTATION GATE TRAINING"
    )
    print("=" * 70)

    # ========================================================
    # LOAD QWEN
    # ========================================================

    print()
    print(
        "Loading Qwen2.5-VL 3B..."
    )

    loaded = load_vlm(
        model_size=ModelSize.SMALL_3B,
        model_family=ModelFamily.QWEN,
    )

    model = loaded.model
    processor = loaded.processor
    adapter = loaded.adapter

    print(
        "Model loaded."
    )

    # ========================================================
    # FREEZE QWEN
    # ========================================================

    for parameter in model.parameters():

        parameter.requires_grad = False

    model.eval()

    print(
        "Qwen parameters frozen."
    )

    # ========================================================
    # CREATE NINE INDEPENDENT GATES
    # ========================================================

    gates = {}

    for head in PROMOTING_HEADS:

        gate = HeadOnlyGate(
            head_dim=128,
            hidden_dim=8,
        )

        # Keep gate parameters in normal floating point.
        gate = gate.to(
            device=model.device,
            dtype=torch.float32,
        )

        gates[head] = gate

    print()
    print(
        f"Created {len(gates)} independent head gates."
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    gate_parameters = []

    for gate in gates.values():

        gate_parameters.extend(
            list(gate.parameters())
        )

    optimizer = torch.optim.Adam(
        gate_parameters,
        lr=LEARNING_RATE,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in gate_parameters
    )

    print(
        f"Trainable gate parameters: "
        f"{trainable_parameters}"
    )

    # Expected:
    #
    # 9 × 1041 = 9369

    # ========================================================
    # DATASET
    # ========================================================

    examples = load_color_examples()

    examples = examples[
        :NUM_EXAMPLES
    ]

    print()
    print(
        f"Training examples: "
        f"{len(examples)}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    (
        start_epoch,
        completed_examples,
        loss_history,
    ) = load_checkpoint(
        gates,
        optimizer,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        epoch_losses = []

        for example_idx, example in enumerate(
            examples
        ):

            # ------------------------------------------------
            # RESUME LOGIC
            # ------------------------------------------------

            if example_idx < completed_examples:

                continue

            print()
            print(
                f"Example "
                f"{example_idx + 1}/{len(examples)}: "
                f"{example.object}"
            )

            prior_color = (
                example.correct_answer[0]
            )

            visual_color = (
                example.incorrect_answer
            )

            print(
                f"  Prior color:  "
                f"{prior_color}"
            )

            print(
                f"  Visual color: "
                f"{visual_color}"
            )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            image = (
                example.counterfact_image.pil_image
            )

            # ------------------------------------------------
            # QUESTION
            # ------------------------------------------------

            question = (
                VMIRunnerBase._get_question(
                    example,
                    GroundingMode.PRIOR,
                )
            )

            inputs = adapter.build_inputs(
                processor=processor,
                image=image,
                question=question,
                device=model.device,
            )

            # ------------------------------------------------
            # TOKEN IDS
            # ------------------------------------------------

            prior_id = get_token_id(
                processor,
                prior_color,
            )

            visual_id = get_token_id(
                processor,
                visual_color,
            )

            # ------------------------------------------------
            # ZERO GRADIENT
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # GATED FORWARD PASS
            # ------------------------------------------------

            with adaptive_head_gate(
                model=model,
                adapter=adapter,
                heads=PROMOTING_HEADS,
                gates=gates,
            ):

                logits = get_logits(
                    model,
                    inputs,
                )

            # ------------------------------------------------
            # CANDIDATE LOGITS
            # ------------------------------------------------

            prior_logit = logits[
                prior_id
            ]

            visual_logit = logits[
                visual_id
            ]

            print(
                f"  Prior logit:  "
                f"{prior_logit.item():.4f}"
            )

            print(
                f"  Visual logit: "
                f"{visual_logit.item():.4f}"
            )

            # ------------------------------------------------
            # RANKING LOSS
            # ------------------------------------------------

            #
            # Desired:
            #
            # visual_logit > prior_logit
            #
            # with margin = 1.
            #

            margin = 1.0

            ranking_loss = F.relu(
                margin
                - visual_logit
                + prior_logit
            )

            # ------------------------------------------------
            # RETENTION LOSS
            # ------------------------------------------------

            #
            # This is only a weak regularizer.
            #
            # The main learning signal is the actual
            # gated Qwen forward pass.
            #

            retention_values = []

            for gate in gates.values():

                gate_dtype = next(
                    gate.parameters()
                ).dtype

                dummy = torch.zeros(
                    1,
                    128,
                    device=model.device,
                    dtype=gate_dtype,
                )

                retention_values.append(
                    (
                        1.0
                        - gate(dummy)
                    ).pow(2).mean()
                )

            retention_loss = torch.stack(
                retention_values
            ).mean()

            # ------------------------------------------------
            # TOTAL LOSS
            # ------------------------------------------------

            loss = (
                ranking_loss
                + RETENTION_WEIGHT
                * retention_loss
            )

            print(
                f"  Ranking loss:  "
                f"{ranking_loss.item():.4f}"
            )

            print(
                f"  Retention loss: "
                f"{retention_loss.item():.4f}"
            )

            print(
                f"  Total loss:     "
                f"{loss.item():.4f}"
            )

            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # GRADIENT DIAGNOSTICS
            # ------------------------------------------------

            total_grad = 0.0

            for gate in gates.values():

                for parameter in gate.parameters():

                    if parameter.grad is not None:

                        total_grad += (
                            parameter.grad
                            .detach()
                            .norm()
                            .item()
                        )

            print(
                f"  Gate gradient norm: "
                f"{total_grad:.6f}"
            )

            # ------------------------------------------------
            # UPDATE GATES
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            loss_value = loss.item()

            epoch_losses.append(
                loss_value
            )

            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "example_index":
                        example_idx + 1,
                    "example":
                        example.object,
                    "prior_color":
                        prior_color,
                    "visual_color":
                        visual_color,
                    "prior_logit":
                        prior_logit.item(),
                    "visual_logit":
                        visual_logit.item(),
                    "ranking_loss":
                        ranking_loss.item(),
                    "retention_loss":
                        retention_loss.item(),
                    "total_loss":
                        loss_value,
                    "gradient_norm":
                        total_grad,
                }
            )

            completed_examples = (
                example_idx + 1
            )

            # =================================================
            # CHECKPOINT AFTER EVERY EXAMPLE
            # =================================================

            save_checkpoint(
                gates=gates,
                optimizer=optimizer,
                epoch=epoch,
                completed_examples=completed_examples,
                loss_history=loss_history,
            )

        # ----------------------------------------------------
        # EPOCH STATISTICS
        # ----------------------------------------------------

        if epoch_losses:

            mean_loss = (
                sum(epoch_losses)
                / len(epoch_losses)
            )

            print()
            print(
                f"Epoch mean loss: "
                f"{mean_loss:.4f}"
            )

        # ----------------------------------------------------
        # Reset example counter for next epoch
        # ----------------------------------------------------

        completed_examples = 0

    # ========================================================
    # FINAL GATE OUTPUTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL GATE OUTPUTS")
    print("=" * 70)

    for head, gate in gates.items():

        gate.eval()

        gate_dtype = next(
            gate.parameters()
        ).dtype

        dummy = torch.zeros(
            1,
            128,
            device=model.device,
            dtype=gate_dtype,
        )

        with torch.no_grad():

            weight = gate(
                dummy
            ).item()

        print(
            f"L{head[0]}_H{head[1]}: "
            f"{weight:.4f}"
        )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    save_checkpoint(
        gates=gates,
        optimizer=optimizer,
        epoch=EPOCHS,
        completed_examples=NUM_EXAMPLES,
        loss_history=loss_history,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
