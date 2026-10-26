import argparse
from pathlib import Path

import torch

from src.data.loader import build_dataset
from src.features.stft import extract_log_power_stft


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate training-only STFT "
            "normalisation statistics."
        )
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

    print(
        f"Training dataset loaded: "
        f"{len(train_dataset)} files"
    )

    total_sum = 0.0
    total_squared_sum = 0.0
    total_values = 0

    for index, sample in enumerate(train_dataset):
        features = extract_log_power_stft(
            waveform=sample["waveform"],
            sample_rate=sample["sample_rate"],
        )

        total_sum += features.sum().item()
        total_squared_sum += (
            features.pow(2).sum().item()
        )
        total_values += features.numel()

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1} / "
                f"{len(train_dataset)} files"
            )

    mean = total_sum / total_values

    variance = (
        total_squared_sum / total_values
    ) - (mean ** 2)

    std = variance ** 0.5

    print(
        "\nFinal training normalisation statistics"
    )
    print("Mean:", mean)
    print("Std:", std)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = (
        results_dir
        / "stft_normalisation.pt"
    )

    torch.save(
        {
            "mean": mean,
            "std": std,
        },
        output_path,
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()