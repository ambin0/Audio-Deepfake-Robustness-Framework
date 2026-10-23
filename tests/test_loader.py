import numpy as np
import soundfile as sf

from src.data.loader import build_dataset


def test_build_train_dataset(tmp_path):
    protocol_dir = (
        tmp_path
        / "ASVspoof2019_LA_cm_protocols"
    )

    protocol_dir.mkdir()

    protocol_file = (
        protocol_dir
        / "ASVspoof2019.LA.cm.train.trn.txt"
    )

    protocol_file.write_text(
        "LA_0001 LA_T_0000001 - - bonafide\n",
        encoding="utf-8",
    )

    audio_dir = (
        tmp_path
        / "ASVspoof2019_LA_train"
        / "flac"
    )

    audio_dir.mkdir(parents=True)

    waveform = np.zeros(16000)

    sf.write(
        audio_dir / "LA_T_0000001.flac",
        waveform,
        16000,
    )

    dataset = build_dataset(
        dataset_root=str(tmp_path),
        partition="train",
    )

    assert len(dataset) == 1

    item = dataset[0]

    assert item["utterance_id"] == "LA_T_0000001"
    assert item["label"] == "bonafide"
    assert item["partition"] == "train"
