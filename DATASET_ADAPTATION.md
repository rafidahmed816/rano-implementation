# Dataset Adaptation Notes (LibriSpeech-Style)

This document summarizes the code changes made to adapt the repository to the dataset layout in `data/` where:

- audio is stored as `.flac`
- each chapter folder contains one transcript file `*.trans.txt`
- transcript lines are formatted as:
  - `<utterance_id> <UPPERCASE TRANSCRIPT...>`

Example chapter:

- `data/train-clean-100/26/496/26-496-0000.flac`
- `data/train-clean-100/26/496/26-496.trans.txt`

## What Was Changed

## 1) Dataset Loader Updated

Updated file: `data.py`

- Added `LibriSpeechDataset` to support:
  - `<root>/<subset>/<speaker>/<chapter>/*.flac`
  - `<root>/<subset>/<speaker>/<chapter>/*.trans.txt`
- Added transcript parsing helper:
  - `_parse_transcript_file(...)`
- Each sample now includes:
  - `mel`
  - `speaker_id`
  - `path`
  - `transcript`
  - `utterance_id`
- Kept backward compatibility by keeping `LibriTTSDataset` as an alias to the new implementation.

## 2) Validation Added

Updated file: `data.py`

- Added `validate_librispeech_layout(...)` to validate:
  - root/subset path resolution
  - transcript file presence per chapter
  - transcript line format
  - audio-to-transcript ID matching
  - transcript-to-audio ID matching

New file: `validate_dataset.py`

- Added standalone CLI validator.
- Exits with non-zero code when validation fails.

### Validation command examples

Validate a subset path directly:

```bash
python validate_dataset.py --root data/train-clean-100
```

Validate from a parent root with subset names:

```bash
python validate_dataset.py --root data --subsets train-clean-100
```

## 3) Training Scripts Adapted

Updated file: `train_stage1.py`
Updated file: `train_stage2.py`

- Added LibriSpeech-compatible CLI options:
  - `--librispeech_root` (alias for `--libritts_root`)
  - `--librispeech_subsets`
  - `--validate_dataset`
  - `--allow_invalid_dataset`
- `build_dataset(...)` now receives validation flags and subset list.

## Usage Examples

Stage 1 (ACG pre-training):

```bash
python train_stage1.py \
  --librispeech_root data \
  --librispeech_subsets train-clean-100 \
  --validate_dataset
```

If your root already points to subset folder:

```bash
python train_stage1.py \
  --librispeech_root data/train-clean-100 \
  --validate_dataset
```

Stage 2 (Rano training):

```bash
python train_stage2.py \
  --librispeech_root data \
  --librispeech_subsets train-clean-100 \
  --validate_dataset
```

## Notes

- `.flac` is now first-class in loader scanning.
- Transcript parsing enforces `utterance_id + text` line structure.
- Validation can be strict (default) or relaxed with `--allow_invalid_dataset`.
