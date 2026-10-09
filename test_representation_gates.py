import torch

from vdpo.representation_gates import (
    HeadOnlyGate,
    ContextOnlyGate,
    HeadContextGate,
)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 70)
    print("REPRESENTATION-BASED GATE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Simulated Qwen representations
    # ---------------------------------------------------------

    batch_size = 20

    # One selected attention head
    head_output = torch.randn(
        batch_size,
        128,
    )

    # Same-token Qwen hidden state
    hidden_state = torch.randn(
        batch_size,
        2048,
    )

    # ---------------------------------------------------------
    # Head-only
    # ---------------------------------------------------------

    head_gate = HeadOnlyGate()

    head_weights = head_gate(
        head_output
    )

    print()
    print("HEAD-ONLY")
    print("-" * 50)

    print(
        "Input shape:",
        tuple(head_output.shape),
    )

    print(
        "Output shape:",
        tuple(head_weights.shape),
    )

    print(
        "Parameters:",
        count_parameters(head_gate),
    )

    print(
        "Min weight:",
        head_weights.min().item(),
    )

    print(
        "Max weight:",
        head_weights.max().item(),
    )

    # ---------------------------------------------------------
    # Context-only
    # ---------------------------------------------------------

    context_gate = ContextOnlyGate()

    context_weights = context_gate(
        hidden_state
    )

    print()
    print("CONTEXT-ONLY")
    print("-" * 50)

    print(
        "Input shape:",
        tuple(hidden_state.shape),
    )

    print(
        "Output shape:",
        tuple(context_weights.shape),
    )

    print(
        "Parameters:",
        count_parameters(context_gate),
    )

    # ---------------------------------------------------------
    # Head + Context
    # ---------------------------------------------------------

    combined_gate = HeadContextGate()

    context_projector = torch.nn.Sequential(
        torch.nn.Linear(2048, 32),
        torch.nn.GELU(),
    )

    projected_context = context_projector(hidden_state)

    combined_weights = combined_gate(   
        head_output,
        projected_context,
    )

    print()
    print("HEAD + CONTEXT")
    print("-" * 50)

    print(
        "Head input:",
        tuple(head_output.shape),
    )

    print(
        "Context input:",
        tuple(hidden_state.shape),
    )

    print(
        "Output shape:",
        tuple(combined_weights.shape),
    )

    print(
        "Parameters:",
        count_parameters(combined_gate),
    )

    # ---------------------------------------------------------
    # Gradient test
    # ---------------------------------------------------------

    loss = (
        head_weights.mean()
        + context_weights.mean()
        + combined_weights.mean()
    )

    loss.backward()

    print()
    print("=" * 70)
    print("GRADIENT CHECK")
    print("=" * 70)

    for name, parameter in head_gate.named_parameters():

        print(
            f"Head-only {name}: "
            f"{parameter.grad.norm().item():.6f}"
        )

    for name, parameter in context_gate.named_parameters():

        print(
            f"Context-only {name}: "
            f"{parameter.grad.norm().item():.6f}"
        )

    for name, parameter in combined_gate.named_parameters():

        print(
            f"Combined {name}: "
            f"{parameter.grad.norm().item():.6f}"
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()