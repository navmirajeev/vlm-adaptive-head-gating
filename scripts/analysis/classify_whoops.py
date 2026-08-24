"""Classify whoops-aha attention heads and MLP layers (PCA on restoration scores).

Reads whoops patching outputs and writes classifications to
``data/classify_attn_heads_whoops.json`` and
``data/classify_mlp_layers_whoops.json`` so the knockout runners can load
them via ``contrast="whoops"``.

Unlike the color task there is no correctness filter: the whoops subset is
already flip-filtered at load time, so every example with patching data
enters the PCA.
"""

import argparse

from scripts.analysis._patching_common import (
    has_patching_data,
    load_patching_results,
)
from scripts.analysis.classify_attn_heads import (
    classify_attn_heads,
)
from scripts.analysis.classify_attn_heads import (
    print_classifications as print_head_classifications,
)
from scripts.analysis.classify_attn_heads import (
    save_classifications as save_head_classifications,
)
from scripts.analysis.classify_mlp_layers import (
    classify_mlp_layers,
)
from scripts.analysis.classify_mlp_layers import (
    print_classifications as print_mlp_classifications,
)
from scripts.analysis.classify_mlp_layers import (
    save_classifications as save_mlp_classifications,
)
from vdpo.settings import settings

WHOOPS_OUTPUT_DIR = settings.output_dir / "whoops"
DATA_DIR = settings.output_dir.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    # Attention heads
    heads_dir = WHOOPS_OUTPUT_DIR / "patching_last_token_attn_heads"
    head_results = [r for r in load_patching_results(heads_dir) if has_patching_data(r)]
    print(f"Loaded {len(head_results)} whoops attn-head patching results.")
    head_cls = classify_attn_heads(head_results, sigma_threshold=args.sigma)
    print_head_classifications(head_cls)
    save_head_classifications(head_cls, DATA_DIR / "classify_attn_heads_whoops.json")

    # MLP layers
    mlp_dir = WHOOPS_OUTPUT_DIR / "patching_last_token_mlp"
    mlp_results = [r for r in load_patching_results(mlp_dir) if has_patching_data(r)]
    print(f"Loaded {len(mlp_results)} whoops MLP patching results.")
    mlp_cls = classify_mlp_layers(mlp_results, sigma_threshold=args.sigma)
    print_mlp_classifications(mlp_cls)
    save_mlp_classifications(mlp_cls, DATA_DIR / "classify_mlp_layers_whoops.json")


if __name__ == "__main__":
    main()
