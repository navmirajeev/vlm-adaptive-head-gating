import torch

from vdpo.types.enums import (
    ModelFamily,
    ModelSize,
)

from vdpo.utils.load_model import load_vlm


loaded = load_vlm(
    model_size=ModelSize.SMALL_3B,
    model_family=ModelFamily.QWEN,
)

model = loaded.model
adapter = loaded.adapter

layers = adapter.get_language_layer_modules(model)

layer_idx = 26
layer = layers[layer_idx]

print("=" * 70)
print(f"QWEN LANGUAGE LAYER {layer_idx}")
print("=" * 70)

print("\nLayer:")
print(layer)

print("\nSelf attention:")
print(layer.self_attn)

print("\nLayerNorm modules:")

for name, module in layer.named_modules():
    if "norm" in name.lower():
        print(
            f"{name}: "
            f"{module.__class__.__name__}"
        )

print("\nDirect layer children:")

for name, module in layer.named_children():
    print(
        f"{name}: "
        f"{module.__class__.__name__}"
    )