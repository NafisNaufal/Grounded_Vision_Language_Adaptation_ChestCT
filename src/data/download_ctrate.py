"""
download_ctrate.py
------------------
Downloads the CT-RATE dataset from HuggingFace.

CT-RATE (ibrahimhamamci/CT-RATE) contains:
  - train/ and valid/ splits of 3D chest CT volumes (NIfTI .nii.gz)
  - radiology_text_reports/ CSV files with paired report text

Usage:
    python src/data/download_ctrate.py --output /data/ct_rate

The script is resumable: already-downloaded files are skipped.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default="/data/ct_rate",
        help="Directory to download CT-RATE into",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, download only this many volumes (for quick testing). "
             "Leave unset to download the full dataset.",
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token if the dataset requires authentication",
    )
    return p.parse_args()


def download_ctrate(output_dir: str, hf_token: str | None = None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CT-RATE → {output_path}")
    print("This may take several hours depending on your connection speed.")
    print("The download is resumable — re-run the script if it is interrupted.\n")

    snapshot_download(
        repo_id="ibrahimhamamci/CT-RATE",
        repo_type="dataset",
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
        token=hf_token,
        ignore_patterns=["*.md", ".gitattributes"],  # skip non-data files
    )

    # Verify the expected top-level structure
    expected = ["train", "valid", "radiology_text_reports"]
    missing = [d for d in expected if not (output_path / d).exists()]
    if missing:
        print(f"\nWarning: expected directories not found: {missing}")
        print("The dataset structure may differ from what was anticipated.")
        print("Check the contents of", output_path)
    else:
        volumes = list(output_path.glob("train/**/*.nii.gz"))
        reports_csv = list(output_path.glob("radiology_text_reports/*.csv"))
        print(f"\nDownload complete.")
        print(f"  Train volumes : {len(volumes)}")
        print(f"  Report CSVs   : {len(reports_csv)}")
        print(f"  Location      : {output_path}")


def main():
    args = parse_args()

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    download_ctrate(args.output, hf_token=args.hf_token)

    if args.subset is not None:
        print(
            f"\nNote: --subset {args.subset} was specified but snapshot_download "
            "downloads everything. Filter during preprocessing instead."
        )


if __name__ == "__main__":
    main()
