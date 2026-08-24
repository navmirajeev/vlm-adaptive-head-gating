"""Run whoops-aha pipeline stages (generalization experiments).

Each stage instantiates the corresponding whoops runner and processes the
flip-filtered example subset. Per-example failures are logged and skipped
(skip-and-continue policy); a stage only fails if it cannot start at all.

Usage:
    PYTHONPATH=src python scripts/run_whoops_pipeline.py --stage res_stream
    PYTHONPATH=src python scripts/run_whoops_pipeline.py --stage attn_heads --samples 2
"""

import argparse
import gc
import sys

import torch
from tqdm.auto import tqdm

from vdpo.types.enums import GroundingMode, ImageVariant, ModelFamily, ModelSize
from vdpo.utils.load_whoops_examples import load_whoops_examples
from vdpo.utils.logger import logger

STAGES = [
    "inference",
    "res_stream",
    "attn_heads",
    "mlp",
    "knockout_attn",
    "knockout_attn_frozen",
    "knockout_mlp",
    "attn_weights",
]


def build_runner(stage: str, model_size: ModelSize, model_family: ModelFamily):
    # Imports deferred so a broken runner import doesn't block other stages.
    if stage == "inference":
        from vdpo.runner.whoops import WhoopsInferenceRunner

        return WhoopsInferenceRunner(model_size, model_family=model_family)
    if stage == "res_stream":
        from vdpo.runner.whoops import WhoopsPatchingLastTokenResStream

        return WhoopsPatchingLastTokenResStream(model_size, model_family=model_family)
    if stage == "attn_heads":
        from vdpo.runner.whoops import WhoopsPatchingLastTokenAttnHeads

        return WhoopsPatchingLastTokenAttnHeads(model_size, model_family=model_family)
    if stage == "mlp":
        from vdpo.runner.whoops import WhoopsPatchingLastTokenMLP

        return WhoopsPatchingLastTokenMLP(model_size, model_family=model_family)
    if stage == "knockout_attn":
        from vdpo.runner.whoops import WhoopsKnockoutAttnHeads

        return WhoopsKnockoutAttnHeads(
            model_size, model_family=model_family, contrast="whoops"
        )
    if stage == "knockout_attn_frozen":
        from vdpo.runner.whoops import WhoopsKnockoutAttnHeads

        # Frozen color circuit (data/classify_attn_heads_frozen_color.json)
        # ablated on the whoops examples: discovery on dataset A, evaluation
        # on dataset B. Outputs land in knockout_attn_heads/frozen_color/.
        return WhoopsKnockoutAttnHeads(
            model_size, model_family=model_family, contrast="frozen_color"
        )
    if stage == "knockout_mlp":
        from vdpo.runner.whoops import WhoopsKnockoutMLP

        return WhoopsKnockoutMLP(
            model_size, model_family=model_family, contrast="whoops"
        )
    if stage == "attn_weights":
        from vdpo.runner.whoops import WhoopsExtractAttentionWeights

        return WhoopsExtractAttentionWeights(model_size, model_family=model_family)
    raise ValueError(f"Unknown stage: {stage}")


def process(runner, stage: str, example) -> None:
    if stage == "inference":
        for grounding_mode in GroundingMode:
            runner.process_example(
                example,
                grounding_mode=grounding_mode,
                image_variant=ImageVariant.COUNTERFACTUAL,
            )
        return
    runner.process_example(example)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, required=True, choices=STAGES)
    parser.add_argument(
        "--model-size",
        type=ModelSize,
        default=ModelSize.SMALL_3B,
        choices=list(ModelSize),
    )
    parser.add_argument(
        "--model-family",
        type=ModelFamily,
        default=ModelFamily.QWEN,
        choices=list(ModelFamily),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=-1,
        help="Limit to first N examples (smoke tests).",
    )
    args = parser.parse_args()

    examples = load_whoops_examples(
        model_family=args.model_family.value,
        model_size=args.model_size.value,
        samples=args.samples,
    )
    if not examples:
        logger.error("No whoops examples loaded; aborting stage.")
        sys.exit(2)

    runner = build_runner(args.stage, args.model_size, args.model_family)

    n_ok, n_fail = 0, 0
    for example in tqdm(examples, desc=f"whoops {args.stage}", unit="ex"):
        try:
            process(runner, args.stage, example)
            n_ok += 1
        except Exception:
            n_fail += 1
            logger.exception(
                f"Example {example.example_id} failed in stage {args.stage}"
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"STAGE_RESULT stage={args.stage} ok={n_ok} fail={n_fail}")
    print(f"STAGE_RESULT stage={args.stage} ok={n_ok} fail={n_fail}", flush=True)
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
