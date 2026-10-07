import numpy as np
import pytest

from src.data.audio import validate_audio


def test_valid_audio():
    waveform = np.zeros(16000)

    validate_audio(
        waveform=waveform,
        sample_rate=16000,
    )


def test_empty_audio_raises_error():
    waveform = np.array([])

    with pytest.raises(ValueError):
        validate_audio(
            waveform=waveform,
            sample_rate=16000,
        )


def test_wrong_sample_rate_raises_error():
    waveform = np.zeros(16000)

    with pytest.raises(ValueError):
        validate_audio(
            waveform=waveform,
            sample_rate=8000,
        )


def test_non_finite_audio_raises_error():
    waveform = np.array([0.0, np.nan, 0.5])

    with pytest.raises(ValueError):
        validate_audio(
            waveform=waveform,
            sample_rate=16000,
        )
