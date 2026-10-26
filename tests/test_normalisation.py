import torch

from src.features.normalisation import normalise_features


def test_normalisation():
    features = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    mean = 2.0
    std = 1.0

    normalised = normalise_features(
        features=features,
        mean=mean,
        std=std,
    )

    expected = torch.tensor(
        [-1.0, 0.0, 1.0]
    )

    assert torch.allclose(
        normalised,
        expected,
    )