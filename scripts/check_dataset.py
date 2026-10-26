import argparse

from src.data.loader import build_dataset
from src.features.normalisation import (
    load_normalisation_stats,
    normalise_features,
)
from src.features.stft import extract_log_power_stft


def main():
    parser = argparse.ArgumentParser(
        description="Check the ASVspoof 2019 LA dataset."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the ASVspoof 2019 LA dataset root.",
    )

    args = parser.parse_args()

    train_dataset = build_dataset(
        dataset_root=args.dataset_root,
        partition="train",
    )

    dev_dataset = build_dataset(
        dataset_root=args.dataset_root,
        partition="dev",
    )

    eval_dataset = build_dataset(
        dataset_root=args.dataset_root,
        partition="eval",
    )

    print("Train samples:", len(train_dataset))
    print("Dev samples:", len(dev_dataset))
    print("Eval samples:", len(eval_dataset))

    sample = train_dataset[0]

    print("\nFirst training example")
    print("Utterance:", sample["utterance_id"])
    print("Label:", sample["label"])
    print("System ID:", sample["system_id"])
    print("Partition:", sample["partition"])
    print("Sample rate:", sample["sample_rate"])
    print("Waveform length:", len(sample["waveform"]))

    features = extract_log_power_stft(
        waveform=sample["waveform"],
        sample_rate=sample["sample_rate"],
    )

    print("\nSTFT features")
    print("Feature shape:", features.shape)
    print("Frequency bins:", features.shape[0])
    print("Time frames:", features.shape[1])
    print(
        "Contains NaN:",
        features.isnan().any().item(),
    )
    print(
        "Contains infinity:",
        features.isinf().any().item(),
    )

    mean, std = load_normalisation_stats(
        "results/stft_normalisation.pt"
    )

    normalised_features = normalise_features(
        features=features,
        mean=mean,
        std=std,
    )

    print("\nNormalised STFT features")
    print("Saved mean:", mean)
    print("Saved std:", std)
    print(
        "Normalised feature mean:",
        normalised_features.mean().item(),
    )
    print(
        "Normalised feature std:",
        normalised_features.std().item(),
    )
    print(
        "Contains NaN:",
        normalised_features.isnan().any().item(),
    )
    print(
        "Contains infinity:",
        normalised_features.isinf().any().item(),
    )


if __name__ == "__main__":
    main()