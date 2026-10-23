from pathlib import Path
from typing import Literal

from src.data.dataset import ASVspoofDataset
from src.data.protocols import load_protocol


Partition = Literal["train", "dev", "eval"]


def build_dataset(
    dataset_root: str,
    partition: Partition,
) -> ASVspoofDataset:
    """
    Build an ASVspoof 2019 Logical Access dataset for a
    specified partition.

    Parameters
    ----------
    dataset_root:
        Path to the ASVspoof2019_LA directory.

    partition:
        One of: "train", "dev", or "eval".

    Returns
    -------
    ASVspoofDataset
        Dataset containing audio and metadata for the
        requested partition.
    """

    root = Path(dataset_root)

    protocol_map = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    if partition not in protocol_map:
        raise ValueError(
            f"Invalid partition: {partition}. "
            "Expected 'train', 'dev', or 'eval'."
        )

    protocol_path = (
        root
        / "ASVspoof2019_LA_cm_protocols"
        / protocol_map[partition]
    )

    records = load_protocol(
        str(protocol_path)
    )

    dataset = ASVspoofDataset(
        records=records,
        dataset_root=str(root),
        partition=partition,
    )

    return dataset
