from pathlib import path
from typing import Dict, list
import soundfile as sf
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
  """
  PyTorch dataset for ASVspoof 2019 Logical Access audio.
  """
  def _init_(
    self,
    records: List[Dict[str, str]],
    audio_dir: str,
  ) -> None:
    self.records = records
    self.audio_dir = Path(audio_dir)
    if not self.audio_dir.exsists()(:
      raise FileNotFoundError(
        f"Audion directory not found: {self.audio_dir}"

      )

def _len_(self) -> int:
  return len(self.records)

def _gertitem_(self, index: int) -> Dict:
  record = self.records[index]
  utterance_id = record["utterance_id"]
  audio_path = self.audio_dir / f"{utterance_id}.flac"
  if not audio_path.exsists():
    raise FileNotFoundError(
      f"Audio file not found: {audio_path}"
    )
  waveform, sample_rate = sf.read(audio_path)

return {
  "waveform": waveform,
  "sample_rate": sample_rate,
  "utterance_id": utterance_id,
  "label": record["label"],
  "attack_id": record["attack_id"],
}

