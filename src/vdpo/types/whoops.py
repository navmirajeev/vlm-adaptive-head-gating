"""Types for the whoops-aha generalization experiments.

Each example pairs one whoops-aha image with two re-captioned questions:
a PRIOR question ("...generally?") probing world knowledge and a VISUAL
question ("...here?") probing the image. The answer tokens are the model's
own top-1 predictions under each prompt (from the 4-bit inference pass),
playing the roles of correct_answer (prior) and incorrect_answer (visual)
from the color task.
"""

from pydantic import BaseModel

from vdpo.types.models import DatasetImage


class DatasetExampleWhoops(BaseModel):
    whoops_aha_index: int
    task: str

    prior_question: str
    visual_question: str

    image: DatasetImage

    # Model's own top-1 under each prompt (4-bit inference, per model).
    prior_token_id: int
    visual_token_id: int
    prior_token_text: str
    visual_token_text: str

    @property
    def example_id(self) -> str:
        return f"whoops_{self.whoops_aha_index:04d}"
