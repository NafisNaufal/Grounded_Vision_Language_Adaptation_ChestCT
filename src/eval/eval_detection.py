"""
eval_detection.py
-----------------
Zero-shot pulmonary nodule detection on LIDC-IDRI using the VISTA3D routing
path in the VILA-M3 / MONAI VLM Agent Framework.

For each LIDC-IDRI scan:
  1. Feed the detection prompt + key axial slices to VILA-M3.
  2. VILA-M3 routes to VISTA3D via the MONAI agent framework, which returns
     segmentation masks and 3D bounding boxes.
  3. Predicted bounding boxes are compared against LIDC-IDRI consensus
     annotations using IoU and mAP.

Metrics:
  - IoU  : mean intersection-over-union over all detected/gt box pairs
  - mAP  : mean Average Precision at IoU thresholds 0.10, 0.25, 0.50

Usage:
    # Baseline (pre-fine-tuning)
    python src/eval/eval_detection.py \
        --config configs/train_config.yaml \
        --eval_json /data/processed/lidc_eval.json \
        --vila_repo ~/projects/VILA-M3_nafis/VLM-Radiology-Agent-Framework

    # Fine-tuned model
    python src/eval/eval_detection.py \
        --config configs/train_config.yaml \
        --eval_json /data/processed/lidc_eval.json \
        --vila_repo ~/projects/VILA-M3_nafis/VLM-Radiology-Agent-Framework \
        --lora_adapter ./checkpoints/lora_adapter_final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


# ── IoU utilities ─────────────────────────────────────────────────────────────

def iou_3d(box_a: list, box_b: list) -> float:
    """
    Compute 3D IoU between two axis-aligned boxes.
    Format: [z0, y0, x0, z1, y1, x1]
    """
    az0, ay0, ax0, az1, ay1, ax1 = box_a
    bz0, by0, bx0, bz1, by1, bx1 = box_b

    iz0 = max(az0, bz0)
    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iz1 = min(az1, bz1)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)

    inter = max(0, iz1 - iz0) * max(0, iy1 - iy0) * max(0, ix1 - ix0)
    vol_a = (az1 - az0) * (ay1 - ay0) * (ax1 - ax0)
    vol_b = (bz1 - bz0) * (by1 - by0) * (bx1 - bx0)
    union = vol_a + vol_b - inter

    return inter / union if union > 0 else 0.0


def match_boxes(
    pred_boxes: list[list],
    gt_boxes: list[list],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Greedy matching of predicted boxes to GT boxes at a given IoU threshold.
    Returns (true_positives, false_positives, false_negatives).
    """
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = iou_3d(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_map(
    all_pred_boxes: list[list[list]],
    all_gt_boxes: list[list[list]],
    iou_thresholds: list[float],
) -> dict[str, float]:
    """Compute mAP and per-threshold AP over all scans."""
    results = {}
    for thr in iou_thresholds:
        total_tp = total_fp = total_fn = 0
        for pred, gt in zip(all_pred_boxes, all_gt_boxes):
            tp, fp, fn = match_boxes(pred, gt, thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        results[f"AP@{thr:.2f}"] = round(precision, 4)
        results[f"Recall@{thr:.2f}"] = round(recall, 4)

    results["mAP"] = round(
        np.mean([v for k, v in results.items() if k.startswith("AP@")]), 4
    )
    return results


# ── VILA-M3 inference via MONAI agent framework ───────────────────────────────

def load_vila_agent(vila_repo: str, model_path: str, lora_adapter: str | None, conv_mode: str):
    """
    Load the VILA-M3 model through the MONAI VLM Agent Framework demo interface.
    This gives us access to the built-in VISTA3D routing.
    """
    vila_repo = str(Path(vila_repo).expanduser())
    if vila_repo not in sys.path:
        sys.path.insert(0, vila_repo)
        sys.path.insert(0, str(Path(vila_repo) / "m3"))
        sys.path.insert(0, str(Path(vila_repo) / "m3" / "demo"))

    try:
        # Try to import the MONAI agent interface
        from demo.experts.expert_registry import ExpertRegistry  # type: ignore
        from demo.vila_m3_agent import VilaM3Agent               # type: ignore

        agent = VilaM3Agent(
            model_path=model_path,
            conv_mode=conv_mode,
            lora_path=lora_adapter,
        )
        return agent, "agent"
    except ImportError:
        # Fall back to direct model loading without expert routing
        print(
            "Warning: MONAI agent framework not importable from the given repo path. "
            "Falling back to direct VILA-M3 inference (no VISTA3D routing)."
        )
        from transformers import AutoProcessor, AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if lora_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter)

        model.eval()
        return (model, processor), "direct"


def run_detection(agent_or_tuple, mode: str, record: dict, processed_root: Path) -> list[list]:
    """
    Run detection inference for one LIDC-IDRI scan.
    Returns a list of predicted bounding boxes [[z0,y0,x0,z1,y1,x1], ...].
    """
    from PIL import Image

    images = []
    for rel_path in record["images"]:
        img_path = processed_root / rel_path
        if img_path.exists():
            images.append(Image.open(img_path).convert("RGB"))

    if not images:
        return []

    prompt = record["conversations"][0]["value"]

    if mode == "agent":
        agent = agent_or_tuple
        try:
            response = agent.run(prompt=prompt, images=images)
            # The agent response should contain structured bounding box data
            # Parse the response — format depends on VILA-M3 output convention
            pred_boxes = parse_agent_boxes(response, record)
        except Exception as e:
            print(f"  Agent error: {e}")
            pred_boxes = []
    else:
        # Direct mode: no VISTA3D, parse text output for box coordinates
        model, processor = agent_or_tuple
        device = next(model.parameters()).device
        inputs = processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        response_text = processor.decode(out[0], skip_special_tokens=True)
        pred_boxes = parse_text_boxes(response_text, record)

    return pred_boxes


def parse_agent_boxes(response, record: dict) -> list[list]:
    """
    Extract bounding boxes from the MONAI agent's structured response.
    VISTA3D returns segmentation masks; we derive bounding boxes from them.
    """
    if hasattr(response, "bounding_boxes"):
        return [list(b) for b in response.bounding_boxes]
    if isinstance(response, dict) and "boxes" in response:
        return [list(b) for b in response["boxes"]]
    # If response is text, fall through to text parser
    if isinstance(response, str):
        return parse_text_boxes(response, record)
    return []


def parse_text_boxes(text: str, record: dict) -> list[list]:
    """
    Parse bounding box coordinates from free-text model output.
    Looks for patterns like "box: [z0, y0, x0, z1, y1, x1]" or
    "nodule at (z0, y0, x0) to (z1, y1, x1)".
    """
    import re

    # Pattern: six consecutive numbers (possibly floats) in brackets or parentheses
    pattern = r"[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]"
    matches = re.findall(pattern, text)
    boxes = []
    for m in matches:
        coords = [float(v) for v in m]
        boxes.append(coords)
    return boxes


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--eval_json", default="/data/processed/lidc_eval.json")
    p.add_argument("--vila_repo",
                   default="~/projects/VILA-M3_nafis/VLM-Radiology-Agent-Framework")
    p.add_argument("--model_path", default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--lora_adapter", default=None,
                   help="Path to LoRA adapter. If None, uses the base model.")
    p.add_argument("--output_json", default="results/detection_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_root = Path(cfg["data"]["processed_root"])
    iou_thresholds = cfg["eval"]["iou_thresholds"]

    with open(args.eval_json) as f:
        eval_records = json.load(f)

    print(f"Evaluation records: {len(eval_records)}")

    # Load VILA-M3 with optional VISTA3D routing
    agent, mode = load_vila_agent(
        vila_repo=args.vila_repo,
        model_path=args.model_path,
        lora_adapter=args.lora_adapter,
        conv_mode=cfg["model"]["conv_mode"],
    )
    print(f"Inference mode: {mode}")

    all_pred_boxes = []
    all_gt_boxes = []
    all_ious = []

    for record in tqdm(eval_records, desc="Running detection"):
        gt_boxes = record["gt_boxes_ijk"]
        pred_boxes = run_detection(agent, mode, record, processed_root)

        all_pred_boxes.append(pred_boxes)
        all_gt_boxes.append(gt_boxes)

        # Per-scan mean IoU (best matching)
        scan_ious = []
        for pred in pred_boxes:
            best = max((iou_3d(pred, gt) for gt in gt_boxes), default=0.0)
            scan_ious.append(best)
        if scan_ious:
            all_ious.append(np.mean(scan_ious))

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    map_results = compute_map(all_pred_boxes, all_gt_boxes, iou_thresholds)

    print("\n── Detection Results ──────────────────────")
    print(f"  Mean IoU : {mean_iou:.4f}")
    for metric, value in map_results.items():
        print(f"  {metric}  : {value:.4f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model_path,
        "lora_adapter": args.lora_adapter,
        "n_scans": len(eval_records),
        "mean_iou": mean_iou,
        "detection_metrics": map_results,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
