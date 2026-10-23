from pathlib import Path
from typing import Dict, List


def load_protocol(protocol_path: str) -> List[Dict[str, str]]:
    """
    Load an ASVspoof 2019 Logical Access countermeasure protocol.

    Expected columns:
    speaker_id audio_file_name unused system_id key
    """

    path = Path(protocol_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Unexpected number of fields on line {line_number}: "
                    f"expected 5, found {len(parts)}"
                )

            (
                speaker_id,
                utterance_id,
                unused,
                system_id,
                label,
            ) = parts

            records.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "system_id": system_id,
                    "label": label,
                }
            )

    return records
