"""
Dataset loaders for VCTK and LibriTTS (Sec. IV-A).

Dataset download instructions
------------------------------
VCTK:
  https://datashare.ed.ac.uk/handle/10283/3443
  Direct: https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
  ~11 GB. Unzip to data/vctk/
  Expected layout after unzip:
    data/vctk/VCTK-Corpus/VCTK-Corpus/wav48/<speaker_id>/*.wav
  Pass --vctk_root data/vctk/VCTK-Corpus/VCTK-Corpus

LibriTTS:
  https://www.openslr.org/60/
  Subsets used: train-clean-100, train-clean-360, train-other-500
  train-clean-100: https://us.openslr.org/resources/60/train-clean-100.tar.gz  (~6.3 GB)
  train-clean-360: https://us.openslr.org/resources/60/train-clean-360.tar.gz  (~23 GB)
  train-other-500: https://us.openslr.org/resources/60/train-other-500.tar.gz  (~30 GB)
  Untar to data/libritts/  — tarball unpacks into a LibriTTS/ subfolder automatically.
  Expected layout:
    data/libritts/LibriTTS/train-clean-100/<speaker_id>/<chapter_id>/*.wav
  Pass --libritts_root data/libritts/LibriTTS

Quick download with wget:
  wget -P data/vctk https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
  unzip data/vctk/VCTK-Corpus-0.92.zip -d data/vctk/
  wget -P data/libritts https://us.openslr.org/resources/60/train-clean-100.tar.gz
  tar -xzf data/libritts/train-clean-100.tar.gz -C data/libritts/
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset

from audio import MelProcessor


class VCTKDataset(Dataset):
    """
    VCTK multi-speaker corpus (100+ speakers, 100+ hours).
    Expects layout: <root>/wav48/<speaker_id>/*.wav
    Pass root = data/vctk/VCTK-Corpus/VCTK-Corpus
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        test_ratio: float = 0.15,
        processor: MelProcessor | None = None,
        max_frames: int = 256,
    ):
        self.processor = processor or MelProcessor()
        self.max_frames = max_frames
        files = sorted(Path(root).rglob("*.flac")) + sorted(Path(root).rglob("*.wav"))
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_ratio))
        self.files = files[:split_idx] if split == "train" else files[split_idx:]
        speaker_names = sorted({f.parent.name for f in self.files})
        self.speaker_ids = {name: i for i, name in enumerate(speaker_names)}
        self.max_decode_retries = 20
        self._decode_failures = 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        current_idx = idx
        for _ in range(self.max_decode_retries):
            path = self.files[current_idx]
            try:
                wav, sr = torchaudio.load(path)
                wav = self.processor.resample(wav.mean(0), sr)  # mono
                mel = self.processor.wav_to_mel(wav)  # (1, 1, F, T)
                mel = self._pad_or_trim(mel.squeeze(0))  # (1, F, T)
                spk = torch.tensor(self.speaker_ids[path.parent.name], dtype=torch.long)
                return {"mel": mel, "speaker_id": spk, "path": str(path)}
            except Exception as exc:
                self._decode_failures += 1
                if self._decode_failures <= 5:
                    warnings.warn(f"Skipping unreadable audio file: {path} ({exc})", stacklevel=2)
                current_idx = random.randint(0, len(self.files) - 1)
        raise RuntimeError(
            f"Failed to decode an audio sample after {self.max_decode_retries} retries in VCTKDataset."
        )

    def _pad_or_trim(self, mel: torch.Tensor) -> torch.Tensor:
        T = mel.shape[-1]
        if T >= self.max_frames:
            start = random.randint(0, T - self.max_frames)
            return mel[..., start : start + self.max_frames]
        return torch.nn.functional.pad(mel, (0, self.max_frames - T))


class LibriTTSDataset(Dataset):
    """
    LibriTTS dataset (500+ hours from audiobooks).
    Expects layout: <root>/<subset>/<speaker_id>/<chapter_id>/*.wav
    """

    def __init__(
        self,
        root: str | Path,
        subsets: list[str] | None = None,
        split: str = "train",
        test_ratio: float = 0.15,
        processor: MelProcessor | None = None,
        max_frames: int = 256,
    ):
        self.processor = processor or MelProcessor()
        self.max_frames = max_frames
        subsets = subsets or ["train-clean-100"]
        files = []
        for subset in subsets:
            files.extend(sorted((Path(root) / subset).rglob("*.wav")))
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_ratio))
        self.files = files[:split_idx] if split == "train" else files[split_idx:]
        speaker_names = sorted({f.parts[-3] for f in self.files})
        self.speaker_ids = {name: i for i, name in enumerate(speaker_names)}
        self.max_decode_retries = 20
        self._decode_failures = 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        current_idx = idx
        for _ in range(self.max_decode_retries):
            path = self.files[current_idx]
            try:
                wav, sr = torchaudio.load(path)
                wav = self.processor.resample(wav.mean(0), sr)
                mel = self.processor.wav_to_mel(wav).squeeze(0)
                mel = self._pad_or_trim(mel)
                spk_id = torch.tensor(self.speaker_ids[path.parts[-3]], dtype=torch.long)
                return {"mel": mel, "speaker_id": spk_id, "path": str(path)}
            except Exception as exc:
                self._decode_failures += 1
                if self._decode_failures <= 5:
                    warnings.warn(f"Skipping unreadable audio file: {path} ({exc})", stacklevel=2)
                current_idx = random.randint(0, len(self.files) - 1)
        raise RuntimeError(
            f"Failed to decode an audio sample after {self.max_decode_retries} retries in LibriTTSDataset."
        )

    def _pad_or_trim(self, mel: torch.Tensor) -> torch.Tensor:
        T = mel.shape[-1]
        if T >= self.max_frames:
            start = random.randint(0, T - self.max_frames)
            return mel[..., start : start + self.max_frames]
        return torch.nn.functional.pad(mel, (0, self.max_frames - T))


def build_dataset(
    vctk_root: str | None = None,
    libritts_root: str | None = None,
    split: str = "train",
    libritts_subsets: list[str] | None = None,
    processor: MelProcessor | None = None,
    max_frames: int = 256,
) -> Dataset:
    """Combine VCTK and LibriTTS into a single ConcatDataset."""
    datasets = []
    if vctk_root:
        datasets.append(VCTKDataset(vctk_root, split, processor=processor, max_frames=max_frames))
    if libritts_root:
        datasets.append(
            LibriTTSDataset(
                libritts_root, libritts_subsets, split, processor=processor, max_frames=max_frames
            )
        )
    if not datasets:
        raise ValueError("At least one of vctk_root or libritts_root must be provided.")
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
