import json
import torch


PROMOTING_HEADS = [
    (26, 0),
    (26, 5),
    (26, 6),
    (27, 1),
    (27, 4),
    (28, 3),
    (31, 3),
    (31, 7),
    (34, 14),
]


def load_gate_features(filename="head_screening_results.json"):
    """
    Load the existing head-screening results and convert
    each example into a 36-dimensional feature vector.

    Feature order for every head:

        prior_norm
        visual_norm
        difference_norm
        cosine
    """

    with open(filename, "r") as f:
        results = json.load(f)

    features = []
    metadata = []

    for result in results:

        example_features = []

        for layer_idx, head_idx in PROMOTING_HEADS:

            key = f"L{layer_idx}_H{head_idx}"

            head = result["heads"][key]

            example_features.extend([
                head["prior_norm"],
                head["visual_norm"],
                head["difference_norm"],
                head["cosine"],
            ])

        features.append(example_features)

        metadata.append({
            "object": result["object"],
            "prior_color": result["prior_color"],
            "visual_color": result["visual_color"],
        })

    features = torch.tensor(
        features,
        dtype=torch.float32,
    )

    return features, metadata