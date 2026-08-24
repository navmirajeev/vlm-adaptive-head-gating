"""Load whoops-aha examples for the generalization pipeline.

Joins three sources:
1. ``whoops_aha_table_final.csv`` (repo root): the re-captioned questions.
2. The HF dataset ``francescortu/whoops-aha``: images, keyed on the dataset's
   ``index`` COLUMN value (which drifts from positional row offset — never
   use ``.select(position)``).
3. ``outputs/whoops_inference/{family}/{size}/{idx:04d}.json``: the 4-bit
   inference outputs providing each prompt's top-1 token (the answer tokens)
   and the flip filter.

Only examples whose top-1 predictions differ between the two prompts are
returned (the conflict subset). Token-length equality of the two prompts was
verified for all 42 pairs (2026-07-11); the patching runners additionally
assert input-shape equality per example as a backstop.
"""

import csv
import io
import json
from pathlib import Path

import datasets

from vdpo.settings import settings
from vdpo.types.whoops import DatasetExampleWhoops
from vdpo.utils.logger import logger

WHOOPS_DATASET = "francescortu/whoops-aha"
TABLE_PATH = settings.output_dir.parent / "whoops_aha_table_final.csv"


def load_whoops_examples(
    model_family: str = "qwen",
    model_size: str = "3B",
    *,
    table_path: Path = TABLE_PATH,
    flips_only: bool = True,
    samples: int = -1,
) -> list[DatasetExampleWhoops]:
    with Path(table_path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    inference_dir = settings.output_dir / "whoops_inference" / model_family / model_size
    if not inference_dir.exists():
        raise FileNotFoundError(
            f"Whoops inference outputs not found: {inference_dir}. "
            f"Run scripts/run_whoops_inference.py first."
        )

    ds = datasets.load_dataset(WHOOPS_DATASET)["train"]
    index_to_pos = {idx: pos for pos, idx in enumerate(ds["index"])}

    examples: list[DatasetExampleWhoops] = []
    skipped_no_flip = 0
    for row in rows:
        idx = int(row["whoops_aha_index"])
        f_inf = inference_dir / f"{idx:04d}.json"
        if not f_inf.exists():
            logger.warning(f"No inference output for whoops idx {idx}, skipping")
            continue
        with f_inf.open() as f:
            inf = json.load(f)

        if flips_only and not inf["top1_differ"]:
            skipped_no_flip += 1
            continue

        if idx not in index_to_pos:
            raise ValueError(f"whoops index {idx} not found in dataset")
        pil_image = ds[index_to_pos[idx]]["image"].convert("RGB")
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")

        examples.append(
            DatasetExampleWhoops(
                whoops_aha_index=idx,
                task=row["task"],
                prior_question=row["prior_question"],
                visual_question=row["counterfactual_question"],
                image={"bytes": buf.getvalue()},
                prior_token_id=inf["prior_raw"]["top1_id"],
                visual_token_id=inf["cf_raw"]["top1_id"],
                prior_token_text=inf["prior_raw"]["top1_text"],
                visual_token_text=inf["cf_raw"]["top1_text"],
            )
        )

    examples.sort(key=lambda e: e.whoops_aha_index)
    if samples > 0:
        examples = examples[:samples]

    logger.info(
        f"Loaded {len(examples)} whoops examples for {model_family}/{model_size} "
        f"(skipped {skipped_no_flip} non-flipping)"
    )
    return examples
