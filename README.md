# Grounded Vision–Language Adaptation on Chest CT

**ICSDG 2025 · Universitas Brawijaya**

> Clinically-oriented zero-shot pulmonary nodule detection and case retrieval
> via expert-routed VLM adaptation (VILA-M3 + VISTA3D) on CT-RATE, evaluated
> on LIDC-IDRI.

**Authors:** Dionisius Seraf Saputra, Nafis Naufal Rahman

---

## Overview

```
CT-RATE (fine-tuning)          LIDC-IDRI (zero-shot eval)
       │                                │
  LoRA fine-tune                  VISTA3D routing
  VILA-M3 8B                      (via MONAI agent)
       │                                │
  Retrieval eval               Detection eval
  Recall@1/5/10                IoU + mAP
```

The core idea: use VILA-M3 as an orchestrator that routes spatial localisation
queries to VISTA3D (a 3D CT expert) while handling case retrieval internally.
LoRA fine-tuning on CT-RATE aligns the model to thoracic clinical language
without requiring bounding box annotations.

---

## Requirements

- Python 3.10
- CUDA 12.2
- 3× NVIDIA L40 (48 GB VRAM) or equivalent
- ~3 TB free disk space for CT-RATE + LIDC-IDRI

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd icsdg

# Clone VILA-M3 alongside (required for MONAI agent framework)
git clone https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework \
    --recursive ~/projects/VILA-M3

# Install dependencies and configure environment
bash setup.sh
source ~/projects/icsdg_venv/bin/activate
```

---

## Run Order

### 1. Download datasets

Start both in separate terminal panes — they run independently and take time.

```bash
# CT-RATE (HuggingFace: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
python src/data/download_ctrate.py --output /data/ct_rate
# If the dataset requires a HuggingFace login:
# python src/data/download_ctrate.py --output /data/ct_rate --hf_token <your_token>

# LIDC-IDRI (~125 GB, TCIA: https://www.cancerimagingarchive.net/collection/lidc-idri/)
# Option A: automated via tcia_utils
python src/data/download_lidc.py --output /data/lidc_idri

# Option B: if you already downloaded via the NBIA Data Retriever manually
python src/data/download_lidc.py \
    --output /data/lidc_idri \
    --dicom_home /path/to/your/dicoms \
    --skip_download
```

### 2. Preprocess

```bash
# CT-RATE → instruction pairs + key axial slices
python src/data/prepare_ctrate.py \
    --ctrate_root /data/ct_rate \
    --output_root /data/processed

# LIDC-IDRI → eval records + key axial slices
python src/data/prepare_lidc.py \
    --lidc_root /data/lidc_idri \
    --output_root /data/processed
```

### 3. Evaluate baseline (pre fine-tuning)

```bash
python src/eval/eval_retrieval.py \
    --model_path MONAI/Llama3-VILA-M3-8B \
    --output_json results/baseline_retrieval.json

python src/eval/eval_detection.py \
    --model_path MONAI/Llama3-VILA-M3-8B \
    --vila_repo ~/projects/VILA-M3 \
    --output_json results/baseline_detection.json
```

### 4. Fine-tune

```bash
# Single GPU
python src/train/finetune_lora.py --config configs/train_config.yaml

# Multi-GPU (all 3 L40s)
torchrun --nproc_per_node=3 src/train/finetune_lora.py \
    --config configs/train_config.yaml
```

### 5. Evaluate fine-tuned model

```bash
python src/eval/eval_retrieval.py \
    --model_path ./checkpoints/lora_adapter_final \
    --is_lora_adapter \
    --output_json results/finetuned_retrieval.json

python src/eval/eval_detection.py \
    --model_path MONAI/Llama3-VILA-M3-8B \
    --lora_adapter ./checkpoints/lora_adapter_final \
    --vila_repo ~/projects/VILA-M3 \
    --output_json results/finetuned_detection.json
```

---

## Configuration

All hyperparameters live in `configs/train_config.yaml`:

| Parameter | Value | Notes |
|---|---|---|
| Model | VILA-M3 8B | `MONAI/Llama3-VILA-M3-8B` |
| LoRA rank | 16 | α = 32 |
| LoRA targets | q_proj, v_proj | attention layers only |
| Learning rate | 1e-4 | cosine annealing |
| Epochs | 3 | |
| Batch size | 4 | 1 per device × 4 grad accumulation |
| CT slices | 16 | uniform axial sampling |
| Hold-out | 10% | CT-RATE retrieval evaluation |

---

## Project Structure

```
icsdg/
├── paper/
│   ├── main.tex               ← LaTeX paper (IOP template)
│   └── figures/               ← place architecture diagram here
├── configs/
│   └── train_config.yaml
├── src/
│   ├── data/
│   │   ├── download_ctrate.py
│   │   ├── download_lidc.py
│   │   ├── prepare_ctrate.py
│   │   └── prepare_lidc.py
│   ├── train/
│   │   └── finetune_lora.py
│   └── eval/
│       ├── eval_retrieval.py
│       └── eval_detection.py
├── requirements.txt
└── setup.sh
```

---

## Citation

```bibtex
@article{saputra2025vilaM3chest,
  title   = {Grounded Vision--Language Adaptation on Chest CT and Reports for
             Clinically-Oriented Zero-Shot Pulmonary Nodule Detection and Case Retrieval},
  author  = {Saputra, Dionisius Seraf and Rahman, Nafis Naufal},
  journal = {Journal of Physics: Conference Series},
  year    = {2025}
}
```

---

## Acknowledgements

Built on [VILA-M3](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework)
by NVIDIA / MONAI. Datasets: [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
· [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/).
