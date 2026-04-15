"""
download_ctrate.py
------------------
Downloads a subset of CT-RATE from HuggingFace.

Official source: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
Paper: arXiv:2403.17834 (Hamamci et al., 2024)

Dataset structure on HuggingFace:
  dataset/train/               ← 3D CT volumes (.nii.gz)
  dataset/valid/               ← validation volumes
  dataset/radiology_text_reports/  ← paired reports (CSV)

Full dataset is ~21 TB. For LoRA fine-tuning, 5000 volumes (~2 TB) is sufficient.

Usage:
    python src/data/download_ctrate.py \
        --output /mnt/nas-hpg9/adhi/data/ct_rate \
        --max_volumes 5000
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",      type=str, default="/mnt/nas-hpg9/adhi/data/ct_rate")
    p.add_argument("--hf_token",    type=str, default=None)
    p.add_argument(
        "--max_volumes", type=int, default=5000,
        help="Max number of train volumes to download. 0 = all (21 TB).",
    )
    return p.parse_args()


def download_files(fs, repo_id, file_list, output_path, token, label=""):
    for i, hf_path in enumerate(file_list, 1):
        # hf_path is like: datasets/ibrahimhamamci/CT-RATE/dataset/train/...
        rel = hf_path.replace(f"datasets/{repo_id}/dataset/", "")
        dest = output_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        tmp = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"dataset/{rel}", token=token,
        )
        shutil.copy(tmp, dest)
        if i % 50 == 0 or i == len(file_list):
            print(f"  {label}{i}/{len(file_list)}")


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    repo_id = "ibrahimhamamci/CT-RATE"
    fs = HfFileSystem(token=token)

    # ── 1. Radiology reports ──────────────────────────────────────────────────
    print("Downloading radiology text reports ...")
    report_files = fs.glob(f"datasets/{repo_id}/dataset/radiology_text_reports/*.csv")
    download_files(fs, repo_id, report_files, output_path, token, label="reports ")
    print(f"  Saved to {output_path / 'radiology_text_reports'}")

    # ── 2. Train volumes ──────────────────────────────────────────────────────
    print("\nListing train volumes (this may take a moment) ...")
    all_train = sorted(fs.glob(f"datasets/{repo_id}/dataset/train/**/*.nii.gz"))
    if args.max_volumes > 0:
        all_train = all_train[:args.max_volumes]
    print(f"Downloading {len(all_train)} train volumes ...")
    download_files(fs, repo_id, all_train, output_path, token, label="train ")

    # ── 3. Validation volumes ─────────────────────────────────────────────────
    print("\nListing validation volumes ...")
    all_valid = sorted(fs.glob(f"datasets/{repo_id}/dataset/valid/**/*.nii.gz"))
    if args.max_volumes > 0:
        n_valid = max(10, len(all_valid) * args.max_volumes // 50000)
        all_valid = all_valid[:n_valid]
    print(f"Downloading {len(all_valid)} validation volumes ...")
    download_files(fs, repo_id, all_valid, output_path, token, label="valid ")

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
