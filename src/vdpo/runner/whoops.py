"""Whoops-aha variants of all runners (generalization experiments).

``WhoopsRunnerMixin`` overrides exactly the example-facing surface of
``VMIRunnerBase`` — question construction, image selection, answer-token
lookup, and output directory — so every runner's intervention logic runs
unchanged. The contrast is prompt-only: PRIOR = "...generally?",
VISUAL = "...here?", same single whoops image for both.

Outputs go to ``outputs/whoops/{runner_name}/{family}/{size}/``.

Knockout runners are instantiated with ``contrast="whoops"`` so that
``load_knockout_targets`` resolves ``data/classify_attn_heads_whoops.json``
and ``data/classify_mlp_layers_whoops.json`` (written by
``scripts/analysis/classify_whoops.py``).
"""

from vdpo.runner.extract_attention_weights import ExtractAttentionWeights
from vdpo.runner.inference import InferenceRunner
from vdpo.runner.knockout_attn_heads import KnockoutAttnHeads
from vdpo.runner.knockout_mlp import KnockoutMLP
from vdpo.runner.patching_last_token_attn_heads import PatchingLastTokenAttnHeads
from vdpo.runner.patching_last_token_mlp import PatchingLastTokenMLP
from vdpo.runner.patching_last_token_res_stream import PatchingLastTokenResStream
from vdpo.settings import settings
from vdpo.types.enums import GroundingMode
from vdpo.types.whoops import DatasetExampleWhoops

ANSWER_SUFFIX = "Answer in a single word."


class WhoopsRunnerMixin:
    """Override example handling; keep all intervention logic unchanged."""

    @staticmethod
    def _get_question(
        example: DatasetExampleWhoops, grounding_mode: GroundingMode
    ) -> str:
        if grounding_mode == GroundingMode.PRIOR:
            return f"{example.prior_question} {ANSWER_SUFFIX}"
        if grounding_mode == GroundingMode.VISUAL:
            return f"{example.visual_question} {ANSWER_SUFFIX}"
        raise ValueError(f"Unknown grounding mode: {grounding_mode}")

    @staticmethod
    def _get_image(example: DatasetExampleWhoops, image_variant):
        # Single whoops image for both groundings; the variant is ignored.
        return example.image.pil_image

    def get_answer_token_ids(self, example: DatasetExampleWhoops):
        # (incorrect, correct) == (visual top-1, prior top-1); see whoops types.
        return {example.visual_token_id}, {example.prior_token_id}

    def get_output_dir(self, contrast_subdir: str | None = None):
        # The knockout runners pass contrast_subdir="whoops"; the whoops
        # prefix is already in the base path, so drop it.
        if contrast_subdir == "whoops":
            contrast_subdir = None
        parts = [settings.output_dir, "whoops", self.runner_name]
        if contrast_subdir is not None:
            parts.append(contrast_subdir)
        parts.extend([self.model_family.value, self.model_size.value])
        path = parts[0]
        for p in parts[1:]:
            path = path / p
        path.mkdir(parents=True, exist_ok=True)
        return path


class WhoopsInferenceRunner(WhoopsRunnerMixin, InferenceRunner):
    pass


class WhoopsPatchingLastTokenResStream(WhoopsRunnerMixin, PatchingLastTokenResStream):
    pass


class WhoopsPatchingLastTokenAttnHeads(WhoopsRunnerMixin, PatchingLastTokenAttnHeads):
    pass


class WhoopsPatchingLastTokenMLP(WhoopsRunnerMixin, PatchingLastTokenMLP):
    pass


class WhoopsKnockoutAttnHeads(WhoopsRunnerMixin, KnockoutAttnHeads):
    pass


class WhoopsKnockoutMLP(WhoopsRunnerMixin, KnockoutMLP):
    pass


class WhoopsExtractAttentionWeights(WhoopsRunnerMixin, ExtractAttentionWeights):
    pass
