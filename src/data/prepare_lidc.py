"""
prepare_lidc.py
---------------
Converts LIDC-IDRI DICOM scans into NIfTI volumes and extracts key axial slices
in the same format used for CT-RATE, ready for zero-shot detection evaluation.

Reads:  /data/lidc_idri/nodule_annotations.json  (written by download_lidc.py)
Writes: /data/processed/lidc_eval.json            (evaluation records)
        /data/processed/slices/lidc_*/             (key axial slices)

Each evaluation record:
    {
        "id": str,
        "volume_id": str,
        "images": [<slice_path>, ...],
        "gt_boxes_ijk": [[z0,y0,x0, z1,y1,x1], ...],   # from LIDC consensus
        "conversations": [{...}]                          # detection prompt only
    }

Usage:
    python src/data/prepare_lidc.py \
        --lidc_root /data/lidc_idri \
        --output_root /data/processed \
        --max_slices 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Slice utilities (same as prepare_ctrate.py) ───────────────────────────────

def window_ct(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def sample_key_slices(vol: np.ndarray, n: int) -> tuple[list[np.ndarray], list[int]]:
    """Return (slices, sampled_z_indices) skipping top/bottom 10%."""
    z = vol.shape[0]
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, n, dtype=int)
    return [vol[i] for i in indices], indices.tolist()


def save_slices(
    slices: list[np.ndarray],
    out_dir: Path,
    volume_id: str,
    target_size: int = 224,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        img = Image.fromarray(sl).convert("RGB")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        rel = f"slices/{volume_id}_{i:02d}.png"
        img.save(out_dir / f"{volume_id}_{i:02d}.png")
        paths.append(rel)
    return paths


# ── DICOM → numpy via pydicom ─────────────────────────────────────────────────

def load_dicom_series(dicom_dir: Path) -> np.ndarray:
    """Load a sorted DICOM series and return a float32 (Z, H, W) array in HU."""
    import pydicom

    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices = []
    for dcm_path in dcm_files:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)
        # Convert to Hounsfield Units
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        slices.append(arr)

    return np.stack(slices, axis=0)  # (Z, H, W)


# ── Consensus mask builder ────────────────────────────────────────────────────

def _build_consensus_mask(
    scan_meta: dict,
    dicom_dir: Path,
    masks_dir: Path,
    volume_id: str,
    vol_shape: tuple,
) -> str | None:
    """
    Build a binary consensus segmentation mask from pylidc nodule annotations.
    A voxel is marked positive if ≥3 radiologists included it in their contour.
    Saves as NIfTI to masks_dir/<volume_id>_gt.nii.gz and returns the path.
    """
    try:
        import pylidc as pl
        import nibabel as nib

        masks_dir.mkdir(parents=True, exist_ok=True)
        out_path = masks_dir / f"{volume_id}_gt.nii.gz"
        if out_path.exists():
            return str(out_path)

        scan = pl.query(pl.Scan).filter(
            pl.Scan.patient_id == scan_meta["patient_id"]
        ).first()
        if scan is None:
            return None

        consensus_mask = np.zeros(vol_shape, dtype=np.uint8)
        for cluster in scan.cluster_annotations(clevel=0.5):
            if len(cluster) < 3:
                continue
            # build_mask returns (mask, cbbox, masks) - use the full-volume mask
            cmask, cbbox, _ = pl.utils.consensus(cluster, clevel=0.5, pad=0)
            slc = (
                slice(cbbox[0].start, cbbox[0].stop),
                slice(cbbox[1].start, cbbox[1].stop),
                slice(cbbox[2].start, cbbox[2].stop),
            )
            # pylidc mask is (x, y, z); our volume is (z, y, x) — transpose
            consensus_mask[slc[2], slc[1], slc[0]] |= cmask.transpose(2, 1, 0)

        nib.save(
            nib.Nifti1Image(consensus_mask, affine=np.eye(4)),
            str(out_path),
        )
        return str(out_path)

    except Exception as e:
        print(f"  Warning: could not build consensus mask for {volume_id}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lidc_root", default="/data/lidc_idri")
    p.add_argument("--output_root", default="/data/processed")
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--slice_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()

    lidc_root = Path(args.lidc_root)
    output_root = Path(args.output_root)
    slices_dir = output_root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    ann_path = lidc_root / "nodule_annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotation index not found at {ann_path}. "
            "Run download_lidc.py first."
        )

    with open(ann_path) as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotated scans from LIDC-IDRI")

    records = []
    dicom_base = lidc_root / "dicoms"

    for scan_meta in tqdm(annotations, desc="Processing LIDC-IDRI"):
        patient_id = scan_meta["patient_id"]
        scan_id = scan_meta["scan_id"]
        volume_id = f"lidc_{patient_id}_{scan_id}"

        # Locate DICOM files for this scan
        patient_dir = dicom_base / patient_id
        if not patient_dir.exists():
            # pylidc stores files under LIDC-IDRI/<patient_id>/<study>/<series>
            candidates = list(dicom_base.glob(f"*{patient_id}*/**/*.dcm"))
            if not candidates:
                print(f"  Warning: DICOM not found for {patient_id}, skipping")
                continue
            dicom_dir = candidates[0].parent
        else:
            series_dirs = [d for d in patient_dir.rglob("*") if d.is_dir() and list(d.glob("*.dcm"))]
            if not series_dirs:
                print(f"  Warning: No DICOM series under {patient_dir}, skipping")
                continue
            dicom_dir = series_dirs[0]

        try:
            vol = load_dicom_series(dicom_dir)
            vol_windowed = window_ct(vol)
            slices, z_indices = sample_key_slices(vol_windowed, args.max_slices)
            slice_paths = save_slices(slices, slices_dir, volume_id, args.slice_size)
        except Exception as e:
            print(f"  Warning: skipping {patient_id}: {e}")
            continue

        # Build consensus segmentation mask from pylidc annotations
        gt_mask_path = _build_consensus_mask(
            scan_meta, dicom_dir, output_root / "masks", volume_id, vol.shape
        )

        records.append(
            {
                "id": volume_id,
                "volume_id": volume_id,
                "patient_id": patient_id,
                "scan_id": scan_id,
                "nii_path": str(dicom_dir),               # full-res path for VISTA3D
                "images": slice_paths,
                "sampled_z_indices": z_indices,
                "volume_shape": list(vol.shape),
                "pixel_spacing": scan_meta["pixel_spacing"],
                "slice_thickness": scan_meta["slice_thickness"],
                "gt_mask_path": gt_mask_path,             # consensus segmentation mask
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\n" * len(slice_paths)
                            + "Identify and localise pulmonary nodules in the "
                            "provided chest CT scan."
                        ),
                    }
                ],
            }
        )

    out_path = output_root / "lidc_eval.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    total_nodules = sum(len(r["gt_boxes_ijk"]) for r in records)
    print(f"\nPreprocessing complete.")
    print(f"  Evaluation records : {len(records)}")
    print(f"  Total gt nodules   : {total_nodules}")
    print(f"  Output             : {out_path}")


if __name__ == "__main__":
    main()
