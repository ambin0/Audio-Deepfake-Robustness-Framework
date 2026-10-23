from pathlib import Path
from typing import Dict, List, Literal

import soundfile as sf
from torch.utils.data import Dataset


Partition = Literal["train", "dev", "eval"]


class ASVspoofDataset(Dataset):
    """
    PyTorch dataset for ASVspoof 2019 Logical Access audio.

    The dataset keeps the official train, development, and
    evaluation partitions separate.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        dataset_root: str,
        partition: Partition,
    ) -> None:
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.partition = partition

        partition_map = {
            "train": "ASVspoof2019_LA_train",
            "dev": "ASVspoof2019_LA_dev",
            "eval": "ASVspoof2019_LA_eval",
        }

        if partition not in partition_map:
            raise ValueError(
                f"Invalid partition: {partition}. "
                "Expected 'train', 'dev', or 'eval'."
            )

        self.audio_dir = (
            self.dataset_root
            / partition_map[partition]
            / "flac"
        )

        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]

        utterance_id = record["utterance_id"]
        audio_path = self.audio_dir / f"{utterance_id}.flac"

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = sf.read(audio_path)

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "utterance_id": utterance_id,
            "label": record["label"],
            "system_id": record["system_id"],
            "partition": self.partition,
        }
