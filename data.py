"""Dataset loaders for LibriSpeech-100 and VCTK (§1.3).

Primary dataset: LibriSpeech train-clean-100 (~100 h, ~251 speakers).
Download from: https://www.openslr.org/12/

Expected layout:
  <root>/<subset>/<speaker_id>/<chapter_id>/*.flac
  <root>/<subset>/<speaker_id>/<chapter_id>/<speaker>-<chapter>.trans.txt

Also supports VCTK for additional speaker diversity.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset, ConcatDataset

from audio import MelProcessor


def _parse_transcript_file(transcript_path: Path) -> dict[str, str]:
    """Parse LibriSpeech chapter transcript file into {utterance_id: text}."""
    mapping: dict[str, str] = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed transcript line in {transcript_path} at line {line_no}: {raw_line!r}"
                )
            utt_id, text = parts
            if not text:
                raise ValueError(
                    f"Missing transcript text in {transcript_path} at line {line_no}: {raw_line!r}"
                )
            mapping[utt_id] = text
    return mapping


def validate_librispeech_layout(
    root: str | Path,
    subsets: list[str] | None = None,
    max_issues_to_show: int = 25,
) -> dict[str, Any]:
    """
    Validate LibriSpeech-style layout:
      <root>/<subset>/<speaker>/<chapter>/*.flac
      <root>/<subset>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    Also supports passing root directly as a subset folder (e.g. data/train-clean-100).
    """
    root_path = Path(root)
    if not root_path.exists():
        return {
            "is_valid": False,
            "total_audio_files": 0,
            "total_transcripts": 0,
            "issues": [f"Root path does not exist: {root_path}"],
            "display_issues": [f"Root path does not exist: {root_path}"],
            "resolved_subset_paths": [],
        }

    subset_paths: list[Path] = []
    if subsets:
        for subset in subsets:
            candidate = root_path / subset
            if candidate.exists() and candidate.is_dir():
                subset_paths.append(candidate)
                continue
            if root_path.name == subset and root_path.is_dir():
                subset_paths.append(root_path)
    else:
        subset_paths = [root_path]

    issues: list[str] = []
    total_audio = 0
    total_text = 0

    if not subset_paths:
        issues.append(
            f"No usable subset paths found under {root_path} for subsets={subsets}. "
            "Pass subsets=None when root already points to a subset directory."
        )

    for subset_path in subset_paths:
        for chapter_dir in sorted(subset_path.glob("*/*")):
            if not chapter_dir.is_dir():
                continue

            trans_files = sorted(chapter_dir.glob("*.trans.txt"))
            if len(trans_files) != 1:
                issues.append(
                    f"{chapter_dir}: expected exactly 1 transcript file, found {len(trans_files)}"
                )
                continue

            trans_path = trans_files[0]
            try:
                transcript_map = _parse_transcript_file(trans_path)
            except Exception as exc:
                issues.append(f"{trans_path}: failed to parse transcript ({exc})")
                continue

            audio_files = sorted(chapter_dir.glob("*.flac")) + sorted(chapter_dir.glob("*.wav"))
            if not audio_files:
                issues.append(f"{chapter_dir}: no .flac or .wav files found")
                continue

            audio_ids = {p.stem for p in audio_files}
            text_ids = set(transcript_map.keys())

            total_audio += len(audio_files)
            total_text += len(transcript_map)

            missing_text = sorted(audio_ids - text_ids)
            missing_audio = sorted(text_ids - audio_ids)

            for utt_id in missing_text[:5]:
                issues.append(f"{chapter_dir}: missing transcript for audio id {utt_id}")
            if len(missing_text) > 5:
                issues.append(
                    f"{chapter_dir}: {len(missing_text) - 5} more audio ids without transcript"
                )

            for utt_id in missing_audio[:5]:
                issues.append(f"{chapter_dir}: missing audio file for transcript id {utt_id}")
            if len(missing_audio) > 5:
                issues.append(
                    f"{chapter_dir}: {len(missing_audio) - 5} more transcript ids without audio"
                )

    return {
        "is_valid": len(issues) == 0,
        "total_audio_files": total_audio,
        "total_transcripts": total_text,
        "issues": issues,
        "display_issues": issues[:max_issues_to_show],
        "resolved_subset_paths": [str(p) for p in subset_paths],
    }


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
                mel = self.processor.wav_to_mel(wav)  # (1, F, T)
                mel = self._pad_or_trim(mel.squeeze(0))  # (F, T)
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


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech-style dataset.
    Expects layout: <root>/<subset>/<speaker_id>/<chapter_id>/*.flac
    with one transcript file per chapter: <speaker_id>-<chapter_id>.trans.txt
    Also supports passing root directly as subset path (e.g. data/train-clean-100).
    """

    def __init__(
        self,
        root: str | Path,
        subsets: list[str] | None = None,
        split: str = "train",
        test_ratio: float = 0.15,
        processor: MelProcessor | None = None,
        max_frames: int = 256,
        validate: bool = False,
        fail_on_validation_error: bool = True,
    ):
        self.processor = processor or MelProcessor()
        self.max_frames = max_frames

        root_path = Path(root)
        subsets = subsets or ["train-clean-100"]

        subset_paths: list[Path] = []
        for subset in subsets:
            candidate = root_path / subset
            if candidate.exists() and candidate.is_dir():
                subset_paths.append(candidate)
            elif root_path.name == subset and root_path.is_dir():
                subset_paths.append(root_path)

        if not subset_paths and root_path.is_dir():
            # root may already point to subset directory
            subset_paths = [root_path]

        self.validation_report = None
        if validate:
            self.validation_report = validate_librispeech_layout(root_path, subsets=subsets)
            if not self.validation_report["is_valid"] and fail_on_validation_error:
                preview = "\n".join(f"- {x}" for x in self.validation_report["display_issues"])
                raise ValueError(
                    "LibriSpeech dataset validation failed:\n"
                    f"{preview}\n"
                    f"Total issues: {len(self.validation_report['issues'])}"
                )

        records: list[dict[str, str]] = []
        for subset_path in subset_paths:
            for chapter_dir in sorted(subset_path.glob("*/*")):
                if not chapter_dir.is_dir():
                    continue

                trans_files = sorted(chapter_dir.glob("*.trans.txt"))
                if not trans_files:
                    continue

                transcript_map = _parse_transcript_file(trans_files[0])
                audio_files = sorted(chapter_dir.glob("*.flac")) + sorted(chapter_dir.glob("*.wav"))
                for audio_path in audio_files:
                    utt_id = audio_path.stem
                    text = transcript_map.get(utt_id)
                    if text is None:
                        continue
                    speaker_id = audio_path.parts[-3]
                    records.append(
                        {
                            "path": str(audio_path),
                            "speaker": speaker_id,
                            "transcript": text,
                        }
                    )

        if not records:
            raise ValueError(
                "No usable LibriSpeech samples found. "
                "Check root path, subset names, and transcript/audio pairing."
            )

        random.seed(42)
        random.shuffle(records)
        split_idx = int(len(records) * (1 - test_ratio))
        self.samples = records[:split_idx] if split == "train" else records[split_idx:]
        speaker_names = sorted({sample["speaker"] for sample in self.samples})
        self.speaker_ids = {name: i for i, name in enumerate(speaker_names)}
        self.max_decode_retries = 20
        self._decode_failures = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        current_idx = idx
        for _ in range(self.max_decode_retries):
            sample = self.samples[current_idx]
            path = Path(sample["path"])
            try:
                wav, sr = torchaudio.load(path)
                wav = self.processor.resample(wav.mean(0), sr)
                mel = self.processor.wav_to_mel(wav).squeeze(0)  # (F, T)
                mel = self._pad_or_trim(mel)
                spk_id = torch.tensor(self.speaker_ids[sample["speaker"]], dtype=torch.long)
                return {
                    "mel": mel,
                    "speaker_id": spk_id,
                    "path": str(path),
                    "transcript": sample["transcript"],
                    "utterance_id": path.stem,
                }
            except Exception as exc:
                self._decode_failures += 1
                if self._decode_failures <= 5:
                    warnings.warn(f"Skipping unreadable audio file: {path} ({exc})", stacklevel=2)
                current_idx = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(
            f"Failed to decode an audio sample after {self.max_decode_retries} retries in LibriSpeechDataset."
        )

    def _pad_or_trim(self, mel: torch.Tensor) -> torch.Tensor:
        T = mel.shape[-1]
        if T >= self.max_frames:
            start = random.randint(0, T - self.max_frames)
            return mel[..., start : start + self.max_frames]
        return torch.nn.functional.pad(mel, (0, self.max_frames - T))


class LibriTTSDataset(LibriSpeechDataset):
    """Backward-compatible alias for previous class name."""


def build_dataset(
    vctk_root: str | None = None,
    libritts_root: str | None = None,
    split: str = "train",
    libritts_subsets: list[str] | None = None,
    processor: MelProcessor | None = None,
    max_frames: int = 256,
    validate_librispeech: bool = False,
    fail_on_validation_error: bool = True,
) -> Dataset:
    """Combine VCTK and LibriSpeech-style datasets into a single ConcatDataset."""
    datasets = []
    if vctk_root:
        datasets.append(VCTKDataset(vctk_root, split, processor=processor, max_frames=max_frames))
    if libritts_root:
        datasets.append(
            LibriSpeechDataset(
                libritts_root,
                libritts_subsets,
                split,
                processor=processor,
                max_frames=max_frames,
                validate=validate_librispeech,
                fail_on_validation_error=fail_on_validation_error,
            )
        )
    if not datasets:
        raise ValueError("At least one of vctk_root or libritts_root must be provided.")
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
