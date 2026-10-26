from pathlib import Path

import torch


def normalise_features(
    features: torch.Tensor,
    mean: float,
    std: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Apply training-derived mean/std normalisation
    to log-power STFT features.
    """

    return (features - mean) / (std + eps)


def load_normalisation_stats(path: str):
    """
    Load saved training-derived normalisation statistics.
    """

    stats_path = Path(path)

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalisation statistics not found: {path}"
        )

    stats = torch.load(
        stats_path,
        weights_only=False,
    )

    return stats["mean"], stats["std"]