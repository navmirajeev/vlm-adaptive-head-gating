"""Lightweight, hook-based interventions for tutorials and quick experiments.

`head_knockout` reproduces the last-token attention-head ablation used by
`KnockoutAttnHeads`, but as a plain forward-hook context manager so it also
works inside `model.generate()` without nnsight.
"""

from collections import defaultdict
from contextlib import contextmanager


@contextmanager
def head_knockout(model, adapter, heads):
    """Zero the o_proj-input contribution of attention heads at the last token.

    Args:
        model: the loaded VLM (a transformers ``PreTrainedModel``).
        adapter: the ``VLMAdapter`` for this model family.
        heads: iterable of ``(layer_idx, head_idx)`` tuples to ablate.

    For a single prompt forward the last position is the final prompt token, so
    this matches the slice ``KnockoutAttnHeads`` zeros. During generation with a
    KV cache each step's o_proj input holds only the new token, so the last
    position is the token being predicted and the ablation persists across the
    whole generation. Zeroing targets the last position of every row, so for a
    batch larger than 1 the final token of each sequence is ablated; the tutorial
    runs at batch size 1. All hooks are removed on exit.
    """
    text_config = adapter.get_text_config(model)
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_heads)

    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_idx, head_idx in heads:
        heads_by_layer[layer_idx].append(head_idx)

    layers = adapter.get_language_layer_modules(model)

    def make_pre_hook(head_idxs):
        def pre_hook(module, args, kwargs):
            hidden = args[0].clone()
            for head_idx in head_idxs:
                start = head_idx * head_dim
                end = start + head_dim
                hidden[..., -1, start:end] = 0
            return (hidden, *args[1:]), kwargs

        return pre_hook

    handles = []
    try:
        for layer_idx, head_idxs in heads_by_layer.items():
            o_proj = adapter.get_o_proj_module(layers[layer_idx])
            handles.append(
                o_proj.register_forward_pre_hook(
                    make_pre_hook(head_idxs), with_kwargs=True
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
@contextmanager
def head_weighting(model, adapter, heads, weight):
    """Scale selected attention-head contributions at the last token.

    Args:
        model: the loaded VLM.
        adapter: the VLMAdapter for this model family.
        heads: iterable of (layer_idx, head_idx) tuples.
        weight: scalar applied to each selected head contribution.

    A weight of 1.0 leaves the heads unchanged.
    A weight of 0.0 completely removes them.
    """

    text_config = adapter.get_text_config(model)
    num_heads = text_config.num_attention_heads
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    heads_by_layer: dict[int, list[int]] = defaultdict(list)

    for layer_idx, head_idx in heads:
        heads_by_layer[layer_idx].append(head_idx)

    layers = adapter.get_language_layer_modules(model)

    def make_pre_hook(head_idxs):
        def pre_hook(module, args, kwargs):
            hidden = args[0].clone()

            for head_idx in head_idxs:
                start = head_idx * head_dim
                end = start + head_dim

                hidden[..., -1, start:end] *= weight

            return (hidden, *args[1:]), kwargs

        return pre_hook

    handles = []

    try:
        for layer_idx, head_idxs in heads_by_layer.items():
            o_proj = adapter.get_o_proj_module(layers[layer_idx])

            handles.append(
                o_proj.register_forward_pre_hook(
                    make_pre_hook(head_idxs),
                    with_kwargs=True,
                )
            )

        yield

    finally:
        for handle in handles:
            handle.remove()

@contextmanager
def adaptive_head_weighting(model, adapter, heads, weights):
    """Apply a different weight to each selected attention head.

    Args:
        model: loaded VLM.
        adapter: VLMAdapter.
        heads: iterable of (layer_idx, head_idx) tuples.
        weights: tensor/list containing one weight per head.

    The weights are applied to the selected heads at the last token.
    """

    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    if len(weights) != len(heads):
        raise ValueError(
            f"Expected {len(heads)} weights, "
            f"but received {len(weights)}."
        )

    heads_by_layer = defaultdict(list)

    for head, weight in zip(heads, weights):
        layer_idx, head_idx = head

        heads_by_layer[layer_idx].append(
            (head_idx, float(weight))
        )

    layers = adapter.get_language_layer_modules(model)

    def make_pre_hook(head_weights):

        def pre_hook(module, args, kwargs):

            hidden = args[0].clone()

            for head_idx, weight in head_weights:

                start = head_idx * head_dim
                end = start + head_dim

                hidden[..., -1, start:end] *= weight

            return (hidden, *args[1:]), kwargs

        return pre_hook

    handles = []

    try:

        for layer_idx, head_weights in heads_by_layer.items():

            o_proj = adapter.get_o_proj_module(
                layers[layer_idx]
            )

            handles.append(
                o_proj.register_forward_pre_hook(
                    make_pre_hook(head_weights),
                    with_kwargs=True,
                )
            )

        yield

    finally:

        for handle in handles:
            handle.remove()


@contextmanager
def adaptive_head_weighting_trainable(
    model,
    adapter,
    heads,
    weights,
):
    """Apply differentiable per-head weights.

    Unlike adaptive_head_weighting(), this version preserves
    the PyTorch computation graph so gradients can flow from
    the model output back into the AdaptiveHeadGate.
    """

    text_config = adapter.get_text_config(model)

    num_heads = text_config.num_attention_heads

    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // num_heads,
    )

    if len(weights) != len(heads):
        raise ValueError(
            f"Expected {len(heads)} weights, "
            f"but received {len(weights)}."
        )

    heads_by_layer = defaultdict(list)

    for head, weight in zip(heads, weights):

        layer_idx, head_idx = head

        heads_by_layer[layer_idx].append(
            (head_idx, weight)
        )

    layers = adapter.get_language_layer_modules(model)

    def make_pre_hook(head_weights):

        def pre_hook(module, args, kwargs):

            hidden = args[0].clone()

            for head_idx, weight in head_weights:

                start = head_idx * head_dim
                end = start + head_dim

                hidden[..., -1, start:end] *= weight

            return (hidden, *args[1:]), kwargs

        return pre_hook

    handles = []

    try:

        for layer_idx, head_weights in heads_by_layer.items():

            o_proj = adapter.get_o_proj_module(
                layers[layer_idx]
            )

            handles.append(
                o_proj.register_forward_pre_hook(
                    make_pre_hook(head_weights),
                    with_kwargs=True,
                )
            )

        yield

    finally:

        for handle in handles:
            handle.remove()

from contextlib import contextmanager

import torch


@contextmanager
def adaptive_head_context_gate(
    model,
    adapter,
    heads,
    gates,
    context_projector,
):
    """
    Adaptive head intervention using:

        head representation + same-token hidden context

    Head:
        128 dimensions

    Context:
        2048 -> 32 dimensions

    Gate:
        160 -> 8 -> 1 -> sigmoid

    Intervention:
        h_tilde = g * h
    """

    language_layers = (
        model.language_model.layers
    )

    # Group selected heads by layer.
    heads_by_layer = {}

    for layer_idx, head_idx in heads:

        if layer_idx not in heads_by_layer:
            heads_by_layer[layer_idx] = []

        heads_by_layer[layer_idx].append(
            head_idx
        )

    hooks = []

    # --------------------------------------------------------
    # Process one transformer layer at a time
    # --------------------------------------------------------

    for layer_idx, layer_heads in heads_by_layer.items():

        layer = language_layers[
            layer_idx
        ]

        attention = layer.self_attn
        o_proj = attention.o_proj

        # ----------------------------------------------------
        # Store same-token hidden state
        # ----------------------------------------------------

        context_storage = {}

        def make_attention_hook(
            storage
        ):

            def attention_pre_hook(
                module,
                args,
            ):

                if not args:
                    return

                hidden_states = args[0]

                # hidden_states:
                #
                # [batch, sequence, 2048]
                #
                # We only need the same token that
                # will later be modified.

                last_token = (
                    hidden_states[:, -1, :]
                )

                storage["hidden"] = (
                    last_token
                )

            return attention_pre_hook

        context_hook = (
            attention.register_forward_pre_hook(
                make_attention_hook(
                    context_storage
                )
            )
        )

        hooks.append(
            context_hook
        )

        # ----------------------------------------------------
        # Modify selected head outputs
        # ----------------------------------------------------

        def make_o_proj_hook(
            selected_heads,
            storage,
        ):

            def o_proj_pre_hook(
                module,
                args,
            ):

                if not args:
                    return args

                hidden = args[0]

                #
                # Expected shape:
                #
                # [batch, sequence,
                #  num_heads * head_dim]
                #

                if "hidden" not in storage:
                    return args

                same_token_hidden = (
                    storage["hidden"]
                )

                # ------------------------------------------------
                # Project 2048 -> 32
                # ------------------------------------------------

                context = (
                    context_projector(
                        same_token_hidden
                    )
                )

                # ------------------------------------------------
                # Work on the last token
                # ------------------------------------------------

                last_token = (
                    hidden[:, -1, :]
                )

                # ------------------------------------------------
                # Apply each selected head gate
                # ------------------------------------------------

                modified = last_token.clone()

                num_heads = (
                    module.in_features
                    // 128
                )

                head_dim = 128

                for head_idx in selected_heads:

                    start = (
                        head_idx
                        * head_dim
                    )

                    end = (
                        start
                        + head_dim
                    )

                    head_output = (
                        last_token[
                            :,
                            start:end
                        ]
                    )

                    gate = gates[
                        (
                            layer_idx,
                            head_idx,
                        )
                    ]

                    # Gate input:
                    #
                    # head:    [B, 128]
                    # context: [B, 32]

                    weight = gate(
                        head_output,
                        context,
                    )

                    modified[
                        :,
                        start:end
                    ] = (
                        head_output
                        * weight
                    )

                # ------------------------------------------------
                # Reconstruct hidden tensor
                # ------------------------------------------------

                new_hidden = (
                    hidden.clone()
                )

                new_hidden[
                    :,
                    -1,
                    :
                ] = modified

                return (
                    new_hidden,
                )

            return o_proj_pre_hook

        o_proj_hook = (
            o_proj.register_forward_pre_hook(
                make_o_proj_hook(
                    layer_heads,
                    context_storage,
                )
            )
        )

        hooks.append(
            o_proj_hook
        )

    try:

        yield

    finally:

        for hook in hooks:
            hook.remove()