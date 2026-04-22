from __future__ import annotations

import argparse
from pathlib import Path

from data import validate_librispeech_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root (either top-level root or a subset folder like data/train-clean-100).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subset names under root, e.g. train-clean-100 train-clean-360.",
    )
    parser.add_argument(
        "--max_issues",
        type=int,
        default=25,
        help="Maximum number of issues printed in summary output.",
    )
    args = parser.parse_args()

    report = validate_librispeech_layout(
        root=Path(args.root),
        subsets=args.subsets,
        max_issues_to_show=args.max_issues,
    )

    print("\nDataset Validation Report")
    print("=" * 60)
    print(f"Valid:              {report['is_valid']}")
    print(f"Subset paths:       {report['resolved_subset_paths']}")
    print(f"Audio files:        {report['total_audio_files']}")
    print(f"Transcript entries: {report['total_transcripts']}")
    print(f"Total issues:       {len(report['issues'])}")

    if report["display_issues"]:
        print("\nSample issues:")
        for issue in report["display_issues"]:
            print(f"- {issue}")

    if not report["is_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
