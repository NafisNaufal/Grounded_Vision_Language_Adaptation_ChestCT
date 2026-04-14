"""
prepare_ctrate.py
-----------------
Converts raw CT-RATE data into instruction-following JSON records
used by the LoRA fine-tuning pipeline.

Each CT volume is processed into two instruction types:

  1. Detection instruction:
       prompt  : "Identify and localise thoracic abnormalities in the
                  provided CT scan."
       response: structured description of findings from the paired report

  2. Retrieval instruction:
       prompt  : "Retrieve the most similar case to: <report excerpt>"
       response: volume identifier (used for contrastive loss)

Output structure:
    /data/processed/
        ctrate_train.json     — training split instruction pairs
        ctrate_holdout.json   — 10% holdout for retrieval evaluation
        slices/               — pre-extracted key axial slices (PNG)

Usage:
    python src/data/prepare_ctrate.py \
        --ctrate_root /data/ct_rate \
        --output_root /data/processed \
        --max_slices 16 \
        --holdout_fraction 0.10
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Radiology report cleaning ─────────────────────────────────────────────────

# Findings section patterns common in CT-RATE reports
_FINDING_RE = re.compile(
    r"(findings?|impression|conclusion)[:\s]*(.*?)(?=findings?|impression|"
    r"conclusion|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_findings(report_text: str) -> str:
    """Return the findings / impression text from a raw radiology report."""
    report_text = report_text.strip()
    matches = _FINDING_RE.findall(report_text)
    if matches:
        # Take the longest match (most likely to be the findings section)
        findings = max(matches, key=lambda m: len(m[1]))[1].strip()
        if len(findings) > 20:
            return findings
    # Fallback: return the whole report
    return report_text


# ── Key slice extraction ───────────────────────────────────────────────────────

def load_volume(nii_path: Path) -> np.ndarray:
    """Load a NIfTI volume and return a float32 numpy array (Z, H, W)."""
    import nibabel as nib

    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)
    # NIfTI axes are typically (H, W, Z) — transpose to (Z, H, W)
    if vol.ndim == 3:
        vol = vol.transpose(2, 0, 1)
    return vol


def window_ct(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    """Apply a lung CT window and normalise to [0, 255] uint8."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    vol = np.clip(vol, lo, hi)
    vol = ((vol - lo) / (hi - lo) * 255).astype(np.uint8)
    return vol


def sample_key_slices(vol: np.ndarray, n: int) -> list[np.ndarray]:
    """Uniformly sample n axial slices, skipping the top and bottom 10%."""
    z = vol.shape[0]
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, n, dtype=int)
    return [vol[i] for i in indices]


def save_slices(
    slices: list[np.ndarray],
    out_dir: Path,
    volume_id: str,
    target_size: int = 224,
) -> list[str]:
    """Save slices as PNG files, return list of relative paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        img = Image.fromarray(sl).convert("RGB")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        rel = f"slices/{volume_id}_{i:02d}.png"
        img.save(out_dir / f"{volume_id}_{i:02d}.png")
        paths.append(rel)
    return paths


# ── Instruction record builders ───────────────────────────────────────────────

def make_detection_record(volume_id: str, slice_paths: list[str], findings: str) -> dict:
    return {
        "id": f"det_{volume_id}",
        "type": "detection",
        "volume_id": volume_id,
        "images": slice_paths,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n" * len(slice_paths)
                    + "Identify and localise thoracic abnormalities in the "
                    "provided CT scan. Describe the findings in clinical terms."
                ),
            },
            {
                "from": "gpt",
                "value": findings,
            },
        ],
    }


def make_retrieval_record(volume_id: str, slice_paths: list[str], findings: str) -> dict:
    # Use a short excerpt (first 200 chars) as the retrieval query
    excerpt = findings[:200].rsplit(" ", 1)[0] + " ..."
    return {
        "id": f"ret_{volume_id}",
        "type": "retrieval",
        "volume_id": volume_id,
        "images": slice_paths,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n" * len(slice_paths)
                    + f"Retrieve the most similar historical case to:\n{excerpt}"
                ),
            },
            {
                "from": "gpt",
                "value": volume_id,   # target label for contrastive loss
            },
        ],
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrate_root", default="/data/ct_rate")
    p.add_argument("--output_root", default="/data/processed")
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--slice_size", type=int, default=224)
    p.add_argument("--holdout_fraction", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total samples (useful for dry runs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_report_index(ctrate_root: Path) -> dict[str, str]:
    """Return {volume_id: report_text} from the CT-RATE CSV files."""
    import pandas as pd

    report_dir = ctrate_root / "radiology_text_reports"
    if not report_dir.exists():
        raise FileNotFoundError(
            f"CT-RATE report directory not found at {report_dir}. "
            "Run download_ctrate.py first."
        )

    dfs = []
    for csv_path in report_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {report_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    # CT-RATE CSVs have columns: VolumeName, Findings_EN (or similar)
    # Normalise column names defensively
    combined.columns = [c.strip().lower() for c in combined.columns]

    vol_col = next(
        (c for c in combined.columns if "volume" in c or "file" in c), None
    )
    rep_col = next(
        (c for c in combined.columns if "finding" in c or "report" in c or "text" in c),
        None,
    )

    if vol_col is None or rep_col is None:
        raise ValueError(
            f"Cannot identify volume/report columns in CT-RATE CSV. "
            f"Available columns: {list(combined.columns)}"
        )

    index = {}
    for _, row in combined.iterrows():
        vid = str(row[vol_col]).strip().replace(".nii.gz", "").replace(".nii", "")
        text = str(row[rep_col]).strip()
        if text and text.lower() != "nan":
            index[vid] = text

    print(f"Loaded {len(index)} report entries from {report_dir}")
    return index


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ctrate_root = Path(args.ctrate_root)
    output_root = Path(args.output_root)
    slices_dir = output_root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    # Find all CT volumes
    nii_paths = sorted(ctrate_root.glob("train/**/*.nii.gz"))
    if not nii_paths:
        raise FileNotFoundError(
            f"No .nii.gz files found under {ctrate_root}/train. "
            "Run download_ctrate.py first."
        )

    print(f"Found {len(nii_paths)} CT volumes in CT-RATE train split")

    if args.max_samples:
        nii_paths = nii_paths[: args.max_samples]
        print(f"Capped to {args.max_samples} samples (--max_samples)")

    # Load report index
    report_index = load_report_index(ctrate_root)

    # Split into train / holdout
    random.shuffle(nii_paths)
    n_holdout = int(len(nii_paths) * args.holdout_fraction)
    holdout_paths = nii_paths[:n_holdout]
    train_paths = nii_paths[n_holdout:]

    print(f"Split: {len(train_paths)} train / {len(holdout_paths)} holdout")

    def process_split(paths: list[Path], split_name: str) -> list[dict]:
        records = []
        for nii_path in tqdm(paths, desc=f"Processing {split_name}"):
            volume_id = nii_path.stem.replace(".nii", "")

            # Match volume_id to report
            report_text = report_index.get(volume_id)
            if report_text is None:
                # Try a looser match (some volume names have extra path components)
                for key in report_index:
                    if key in volume_id or volume_id in key:
                        report_text = report_index[key]
                        break

            if report_text is None:
                continue  # no paired report — skip

            findings = extract_findings(report_text)

            # Load and preprocess CT volume
            try:
                vol = load_volume(nii_path)
                vol = window_ct(vol)
                slices = sample_key_slices(vol, args.max_slices)
                slice_paths = save_slices(slices, slices_dir, volume_id, args.slice_size)
            except Exception as e:
                print(f"  Warning: skipping {nii_path.name}: {e}")
                continue

            records.append(make_detection_record(volume_id, slice_paths, findings))
            records.append(make_retrieval_record(volume_id, slice_paths, findings))

        return records

    train_records = process_split(train_paths, "train")
    holdout_records = process_split(holdout_paths, "holdout")

    # Write JSON files
    train_out = output_root / "ctrate_train.json"
    holdout_out = output_root / "ctrate_holdout.json"

    with open(train_out, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(holdout_out, "w") as f:
        json.dump(holdout_records, f, indent=2)

    print(f"\nPreprocessing complete.")
    print(f"  Train records   : {len(train_records)} ({len(train_records)//2} volumes × 2 tasks)")
    print(f"  Holdout records : {len(holdout_records)}")
    print(f"  Train JSON      : {train_out}")
    print(f"  Holdout JSON    : {holdout_out}")
    print(f"  Slices          : {slices_dir}")


if __name__ == "__main__":
    main()
