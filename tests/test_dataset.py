import numpy as np
import pytest
import soundfile as sf

from src.data.dataset import ASVspoofDataset


def test_train_partition_loads_audio(tmp_path):
    audio_dir = (
        tmp_path
        / "ASVspoof2019_LA_train"
        / "flac"
    )

    audio_dir.mkdir(parents=True)

    waveform = np.zeros(16000)

    audio_path = audio_dir / "LA_T_0000001.flac"

    sf.write(
        audio_path,
        waveform,
        16000,
    )

    records = [
        {
            "speaker_id": "LA_0001",
            "utterance_id": "LA_T_0000001",
            "system_id": "-",
            "label": "bonafide",
        }
    ]

    dataset = ASVspoofDataset(
        records=records,
        dataset_root=str(tmp_path),
        partition="train",
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["utterance_id"] == "LA_T_0000001"
    assert item["label"] == "bonafide"
    assert item["partition"] == "train"
    assert item["sample_rate"] == 16000


def test_invalid_partition_raises_error(tmp_path):
    records = []

    with pytest.raises(ValueError):
        ASVspoofDataset(
            records=records,
            dataset_root=str(tmp_path),
            partition="invalid",
        )