"""
download_ctrate.py
------------------
Downloads a subset of CT-RATE from HuggingFace.

Official source: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
Paper: arXiv:2403.17834 (Hamamci et al., 2024)

CT-RATE contains:
  - 3D chest CT volumes (NIfTI .nii.gz)
  - Paired radiology reports (CSV)

Full dataset is ~21 TB. For LoRA fine-tuning, 500–1000 volumes is sufficient.
The radiology_text_reports CSV (~few MB) is always downloaded in full.

Usage:
    # Download 500 volumes (recommended for fine-tuning, ~200 GB)
    python src/data/download_ctrate.py --output /data/ct_rate --max_volumes 500

    # Download 1000 volumes (~400 GB)
    python src/data/download_ctrate.py --output /data/ct_rate --max_volumes 1000

    # Download everything (21 TB, not recommended)
    python src/data/download_ctrate.py --output /data/ct_rate --max_volumes 0
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",      type=str, default="/data/ct_rate")
    p.add_argument("--hf_token",    type=str, default=None)
    p.add_argument(
        "--max_volumes", type=int, default=500,
        help="Max number of train volumes to download. 0 = all (21 TB).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    repo_id = "ibrahimhamamci/CT-RATE"
    fs = HfFileSystem(token=token)

    # ── 1. Always download radiology reports (small, ~few MB) ────────────────
    print("Downloading radiology text reports ...")
    report_files = fs.glob(f"datasets/{repo_id}/radiology_text_reports/*.csv")
    for rf in report_files:
        rel = rf.replace(f"datasets/{repo_id}/", "")
        dest = output_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = hf_hub_download(
                repo_id=repo_id, repo_type="dataset",
                filename=rel, token=token,
            )
            shutil.copy(tmp, dest)
    print(f"  Reports saved to {output_path / 'radiology_text_reports'}")

    # ── 2. Download train volumes up to --max_volumes ─────────────────────────
    all_volumes = sorted(fs.glob(f"datasets/{repo_id}/train/**/*.nii.gz"))
    if args.max_volumes > 0:
        all_volumes = all_volumes[: args.max_volumes]
        print(f"\nDownloading {len(all_volumes)} train volumes "
              f"(--max_volumes {args.max_volumes}) ...")
    else:
        print(f"\nDownloading ALL {len(all_volumes)} train volumes (21 TB) ...")

    for i, vf in enumerate(all_volumes, 1):
        rel = vf.replace(f"datasets/{repo_id}/", "")
        dest = output_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        tmp = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=rel, token=token,
        )
        shutil.copy(tmp, dest)
        if i % 50 == 0 or i == len(all_volumes):
            print(f"  {i}/{len(all_volumes)} volumes downloaded")

    # ── 3. Download validation volumes (small split, used for retrieval eval) ─
    valid_files = sorted(fs.glob(f"datasets/{repo_id}/valid/**/*.nii.gz"))
    # Cap at same proportion as train
    if args.max_volumes > 0:
        n_valid = max(10, len(valid_files) * args.max_volumes // 50000)
        valid_files = valid_files[:n_valid]
    print(f"\nDownloading {len(valid_files)} validation volumes ...")
    for vf in valid_files:
        rel = vf.replace(f"datasets/{repo_id}/", "")
        dest = output_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = hf_hub_download(
                repo_id=repo_id, repo_type="dataset",
                filename=rel, token=token,
            )
            shutil.copy(tmp, dest)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_train = len(list((output_path / "train").glob("**/*.nii.gz")))
    n_valid = len(list((output_path / "valid").glob("**/*.nii.gz")))
    n_rep   = len(list((output_path / "radiology_text_reports").glob("*.csv")))
    print(f"\nDownload complete.")
    print(f"  Train volumes      : {n_train}")
    print(f"  Validation volumes : {n_valid}")
    print(f"  Report CSVs        : {n_rep}")
    print(f"  Location           : {output_path}")


if __name__ == "__main__":
    main()
