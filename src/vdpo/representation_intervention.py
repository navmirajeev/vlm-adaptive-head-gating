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
    Apply learned per-head gates to selected attention-head
    representations at the last token.

        h_i -> g_i(h_i) -> g_i * h_i

    Qwen remains frozen. Only the gate networks are trainable.
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