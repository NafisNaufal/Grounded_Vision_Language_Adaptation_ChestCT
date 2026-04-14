"""
download_lidc.py
----------------
Downloads and indexes the LIDC-IDRI dataset via pylidc.

pylidc is a Python API that wraps the LIDC-IDRI dataset. On first run it
downloads the DICOM files from TCIA (~125 GB). Subsequent runs use the local
cache.

Usage:
    python src/data/download_lidc.py --output /data/lidc_idri

After download, this script also writes a single annotations JSON file:
    /data/lidc_idri/nodule_annotations.json

Each entry in that file is one consensus nodule cluster:
    {
        "scan_id": str,
        "patient_id": str,
        "slice_thickness": float,
        "pixel_spacing": [float, float],
        "nodules": [
            {
                "nodule_id": int,
                "bbox_ijk":  [z_min, y_min, x_min, z_max, y_max, x_max],
                "malignancy": float,       # mean radiologist malignancy score
                "diameter_mm": float
            },
            ...
        ]
    }
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default="/data/lidc_idri",
        help="Directory where LIDC-IDRI DICOM files will be stored",
    )
    p.add_argument(
        "--dicom_home",
        type=str,
        default=None,
        help="If DICOM files are already downloaded, pass the path here "
             "so pylidc does not re-download them.",
    )
    return p.parse_args()


def configure_pylidc(dicom_home: str) -> None:
    """Write a pylidc config so it knows where to look for DICOM files."""
    import configparser

    config = configparser.ConfigParser()
    config["pylidc"] = {"dicom_path": dicom_home}

    cfg_path = Path.home() / ".pylidcrc"
    with open(cfg_path, "w") as f:
        config.write(f)
    print(f"pylidc configured: dicom_path={dicom_home}")


def build_annotation_index(output_dir: Path) -> None:
    try:
        import pylidc as pl
    except ImportError:
        raise ImportError(
            "pylidc is not installed. Run: pip install pylidc"
        )

    print("Building nodule annotation index from LIDC-IDRI ...")
    print("(This will download DICOM files from TCIA if not already present — ~125 GB)\n")

    records = []
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} CT scans in LIDC-IDRI\n")

    for scan in scans:
        try:
            # Cluster annotations with ≥3 radiologist markings (consensus nodules)
            nodule_clusters = scan.cluster_annotations(clevel=0.5)
        except Exception as e:
            print(f"  Skipping scan {scan.patient_id}: {e}")
            continue

        nodules = []
        for i, cluster in enumerate(nodule_clusters):
            if len(cluster) < 3:
                # Fewer than 3 radiologists marked this — skip low-confidence
                continue

            # Build consensus bounding box from all annotations in the cluster
            all_bboxes = np.array([ann.bbox_matrix() for ann in cluster])
            bbox_min = all_bboxes[:, :, 0].min(axis=0).tolist()  # [z, y, x]
            bbox_max = all_bboxes[:, :, 1].max(axis=0).tolist()

            malignancy_scores = [
                ann.malignancy for ann in cluster if ann.malignancy is not None
            ]
            mean_malignancy = float(np.mean(malignancy_scores)) if malignancy_scores else None

            diameters = [ann.diameter for ann in cluster if ann.diameter is not None]
            mean_diameter = float(np.mean(diameters)) if diameters else None

            nodules.append(
                {
                    "nodule_id": i,
                    "bbox_ijk": bbox_min + bbox_max,   # [z0,y0,x0, z1,y1,x1]
                    "malignancy": mean_malignancy,
                    "diameter_mm": mean_diameter,
                }
            )

        if not nodules:
            continue

        records.append(
            {
                "scan_id": str(scan.id),
                "patient_id": scan.patient_id,
                "slice_thickness": scan.slice_thickness,
                "pixel_spacing": [scan.pixel_spacing, scan.pixel_spacing],
                "nodules": nodules,
            }
        )

    out_path = output_dir / "nodule_annotations.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    total_nodules = sum(len(r["nodules"]) for r in records)
    print(f"\nAnnotation index written to {out_path}")
    print(f"  Scans with ≥1 consensus nodule : {len(records)}")
    print(f"  Total consensus nodule clusters : {total_nodules}")


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_home = args.dicom_home or str(output_dir / "dicoms")
    Path(dicom_home).mkdir(parents=True, exist_ok=True)

    configure_pylidc(dicom_home)
    build_annotation_index(output_dir)


if __name__ == "__main__":
    main()
