from collections import defaultdict
from contextlib import contextmanager

import torch


@contextmanager
def adaptive_head_gate(
    model,
    adapter,
    heads,
    gates,
):
    """
    Apply learned per-head gates based only on the selected
    attention-head representation.

        h_i -> g_i(h_i) -> g_i * h_i

    Option 1.
    """

    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads

    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    heads_by_layer = defaultdict(list)

    for layer_idx, head_idx in heads:
        heads_by_layer[layer_idx].append(head_idx)

    layers = adapter.get_language_layer_modules(model)

    handles = []

    def make_pre_hook(layer_idx, head_idxs):

        def pre_hook(module, args, kwargs):

            hidden = args[0]

            parts = []

            for head_idx in range(num_heads):

                start = head_idx * head_dim
                end = start + head_dim

                head_output = hidden[..., -1, start:end]

                if head_idx in head_idxs:

                    gate = gates[
                        (layer_idx, head_idx)
                    ]

                    gate = gate.to(
                        device=head_output.device,
                        dtype=head_output.dtype,
                    )

                    gate_input = head_output.to(
                        dtype=next(gate.parameters()).dtype
                    )

                    weight = gate(gate_input)

                    weight = weight.to(
                        dtype=head_output.dtype
                    )

                    while weight.dim() < head_output.dim():
                        weight = weight.unsqueeze(-1)

                    modified_head = weight * head_output

                else:
                    modified_head = head_output

                parts.append(modified_head)

            new_last_token = torch.cat(parts, dim=-1)

            hidden = torch.cat(
                [
                    hidden[..., :-1, :],
                    new_last_token.unsqueeze(-2),
                ],
                dim=-2,
            )

            return (hidden, *args[1:]), kwargs

        return pre_hook

    try:

        for layer_idx, head_idxs in heads_by_layer.items():

            o_proj = adapter.get_o_proj_module(
                layers[layer_idx]
            )

            handle = o_proj.register_forward_pre_hook(
                make_pre_hook(
                    layer_idx,
                    head_idxs,
                ),
                with_kwargs=True,
            )

            handles.append(handle)

        yield

    finally:

        for handle in handles:
            handle.remove()


@contextmanager
def adaptive_head_context_gate(
    model,
    adapter,
    heads,
    gates,
    context_projector,
):
    """
    Apply learned per-head gates based on:

        1. selected attention-head representation
        2. same-token transformer context

    Head representation:
        h_i ∈ R^128

    Context representation:
        x_t ∈ R^2048

    Context projection:
        2048 -> 32

    Gate input:
        [h_i ; context] ∈ R^160

    Gate:
        160 -> 8 -> 1 -> sigmoid

    Intervention:

        h_i -> g_i(h_i, context) * h_i

    Qwen remains frozen.
    Only the gate networks and context projector are trainable.
    """

    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads

    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    heads_by_layer = defaultdict(list)

    for layer_idx, head_idx in heads:
        heads_by_layer[layer_idx].append(head_idx)

    layers = adapter.get_language_layer_modules(model)

    handles = []

    # Stores the same-token representation for each layer.
    layer_contexts = {}

    def make_context_hook(layer_idx):

        def context_hook(module, args, output):

            # input_layernorm output has shape:
            # [batch, sequence, hidden_dim]
            #
            # We only need the final token.
            context = output[..., -1, :]

            layer_contexts[layer_idx] = context

            return output

        return context_hook

    def make_pre_hook(layer_idx, head_idxs):

        def pre_hook(module, args, kwargs):

            hidden = args[0]

            # Context captured from input_layernorm of this layer.
            context = layer_contexts[layer_idx]

            # Project:
            # 2048 -> 32
            projected_context = context_projector(
                context.to(
                    dtype=next(context_projector.parameters()).dtype
                )
            )

            parts = []

            for head_idx in range(num_heads):

                start = head_idx * head_dim
                end = start + head_dim

                head_output = hidden[..., -1, start:end]

                if head_idx in head_idxs:

                    gate = gates[
                        (layer_idx, head_idx)
                    ]

                    gate = gate.to(
                        device=head_output.device,
                        dtype=head_output.dtype,
                    )

                    # Gate input contains:
                    # 128-dimensional head representation
                    # + 32-dimensional context
                    gate_head_input = head_output.to(
                        dtype=next(gate.parameters()).dtype
                    )

                    gate_context_input = projected_context.to(
                        dtype=gate_head_input.dtype
                    )

                    weight = gate(
                        gate_head_input,
                        gate_context_input,
                    )

                    weight = weight.to(
                        dtype=head_output.dtype
                    )

                    while weight.dim() < head_output.dim():
                        weight = weight.unsqueeze(-1)

                    modified_head = weight * head_output

                else:

                    modified_head = head_output

                parts.append(modified_head)

            new_last_token = torch.cat(
                parts,
                dim=-1,
            )

            hidden = torch.cat(
                [
                    hidden[..., :-1, :],
                    new_last_token.unsqueeze(-2),
                ],
                dim=-2,
            )

            return (hidden, *args[1:]), kwargs

        return pre_hook

    try:

        for layer_idx, head_idxs in heads_by_layer.items():

            layer = layers[layer_idx]

            # Capture the same-token representation before
            # this layer's self-attention.
            input_layernorm = layer.input_layernorm

            context_handle = input_layernorm.register_forward_hook(
                make_context_hook(layer_idx)
            )

            handles.append(context_handle)

            # Modify selected attention heads immediately
            # before their o_proj.
            o_proj = adapter.get_o_proj_module(layer)

            gate_handle = o_proj.register_forward_pre_hook(
                make_pre_hook(
                    layer_idx,
                    head_idxs,
                ),
                with_kwargs=True,
            )

            handles.append(gate_handle)

        yield

    finally:

        for handle in handles:
            handle.remove()

        layer_contexts.clear()