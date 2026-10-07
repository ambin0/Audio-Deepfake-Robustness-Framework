from src.data.protocols import load_protocol


def test_load_protocol(tmp_path):
    protocol_file = tmp_path / "protocol.txt"

    protocol_file.write_text(
        "LA_0079 LA_T_1138215 - - bonafide\n"
        "LA_0080 LA_T_1000137 - A01 spoof\n",
        encoding="utf-8",
    )

    records = load_protocol(str(protocol_file))

    assert len(records) == 2

    assert records[0]["speaker_id"] == "LA_0079"
    assert records[0]["utterance_id"] == "LA_T_1138215"
    assert records[0]["label"] == "bonafide"

    assert records[1]["attack_id"] == "A01"
    assert records[1]["label"] == "spoof"
