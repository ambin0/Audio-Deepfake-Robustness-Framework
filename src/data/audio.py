import numpy as np


def validate_audio(
    waveform: np.ndarray,
    sample_rate: int,
    expected_sample_rate: int = 16000,
) -> None:
    """
    Validate basic properties of an audio waveform.
    """

    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Unexpected sample rate: {sample_rate}. "
            f"Expected {expected_sample_rate}."
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            "Audio waveform contains NaN or infinite values."
        )
