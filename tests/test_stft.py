import numpy as np 
import pytest
import torch

from src.features.stft import extract_log_power_stft

def test_stft_output_is_tensor():
  waveform = np.zeros(16000)
  features = extract_log_power_stft(
   waveform,
   sample_rate=16000,
  )
  assert isinstance(features, torch.Tensor)

def test_stft_has_expected_frequency_bins():
  waveform = np.zeros(16000)
  features = extract_log_power_stft(
  waveform,
  sample_rate=16000,
  n_fft=512,
  )
  assert features.shape[0] == 257

def rest_stft_contains_only_finite_values():
  waveform = np.randn(16000)
  features = extract_log_power_stft(
  waveform,
  sample_rate=16000,
  )
  assert torch.isfinite(features).all()

def test_empty_waveform_raises_error():
  waveform = np.array([])
  with pytest.raises(ValueError):
    extract_log_power_stft(
      waveform,
      sample_rate=16000
    )

def test_multichannel_waveform_raises_errir():
  waveform = np.zeros((2, 16000))
  with pytest.raises(ValueError):
    extract_log_power_stft(
      waveform,
      sample_rate=16000,
    )
