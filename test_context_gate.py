import json
import os

import torch
import torch.nn.functional as F

from vdpo.representation_gates import HeadContextGate
from vdpo.representation_intervention import adaptive_head_context_gate

from vdpo.types.enums import (
    ModelFamily,
    ModelSize,
    GroundingMode,
)

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
NUM_EXAMPLES = 2
EPOCHS = 1
LEARNING_RATE = 1e-3

CHECKPOINT_JSON = "training_context_gate_checkpoint.json"
GATES_FILE = "trained_head_context_gates.pt"

loaded = load_vlm(
    model_size=ModelSize.SMALL_3B,
    model_family=ModelFamily.QWEN,
)

model = loaded.model
processor = loaded.processor
adapter = loaded.adapter

for parameter in model.parameters():
    parameter.requires_grad = False

model.eval()

context_projector = torch.nn.Sequential(
    torch.nn.Linear(
        2048,
        32,
    ),
    torch.nn.GELU(),
).to(model.device)

gates = {}

for head in PROMOTING_HEADS:

    gates[head] = HeadContextGate(
        head_dim=128,
        context_dim=32,
        hidden_dim=8,
    ).to(model.device)

print(
    f"Created {len(gates)} contextual head gates."
)

print(
    "Context projector: 2048 -> 32"
)

all_gate_parameters = []

for gate in gates.values():

    all_gate_parameters.extend(
        list(gate.parameters())
    )

all_gate_parameters.extend(
    list(context_projector.parameters())
)

optimizer = torch.optim.Adam(
    all_gate_parameters,
    lr=LEARNING_RATE,
)

