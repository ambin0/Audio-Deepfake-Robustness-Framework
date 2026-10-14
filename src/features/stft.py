import torch
def extract_log_power_stft(
  waveform,
  sample_rate: int = 16000,
  frame_length_ms: int = 20,
  frame_shift_ms: int = 10,
  n_fft: int = 512,
  eps: float = 1e-10,
) -> torch. Tensor:
  """
  convert a mono audio waveform into a log-power STFFT spectrogram.

  Parameters
  ____
  waveform:
  one-dimensional audio waveform.

  sample_rate:
  Audio sampling rate in Hz.

  Frame_length_ms: 
  Length of each analysis frame in miliseconds.

  frame_shift_ms:
  Distance between consecutive frames in millisecondds.

  n_fft:
  FFT size.

  eps:
  Small value used to prevent log(0)

  returns
  ----
  torch.Tensor
  log-power STFT spectrogram with shape
  [frequency_bins, time_frames].
  """
  waveform = torch.as_tensor(
      waveform,
      dtype=torch.float32
  )

if waveform.ndim != 1:
  raise ValueError(
    "Expected a one dimensional mono waveform"
  )

if waveform.numel() == 0:
  raise ValueError(
    "waveform must not be empty"
  )

win_length = int(
  sample_rate * frame_length_ms / 1000
)

hop_length = int(
  sample_rate * frame_shift_ms / 1000

  window = torch.hamming_window(
        win_length,
        periodic=True,
    )

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )

    power_spectrogram = stft.abs().pow(2)

    log_power_spectrogram = torch.log(
        power_spectrogram + eps
    )

    return log_power_spectrogram
