"""
eval_retrieval.py
-----------------
Evaluates text-to-image case retrieval on the CT-RATE hold-out split.

Method:
  1. Build an embedding index from all hold-out CT volumes using the
     fine-tuned (or baseline) VILA-M3.
  2. For each hold-out volume, use the paired report excerpt as the
     text query to retrieve the top-K most similar volumes.
  3. Compute Recall@K for K ∈ {1, 5, 10} — a hit is when the correct
     volume appears in the top-K results.

Usage:
    # Baseline (no fine-tuning)
    python src/eval/eval_retrieval.py \
        --config configs/train_config.yaml \
        --holdout_json /data/processed/ctrate_holdout.json \
        --model_path MONAI/Llama3-VILA-M3-8B

    # Fine-tuned model
    python src/eval/eval_retrieval.py \
        --config configs/train_config.yaml \
        --holdout_json /data/processed/ctrate_holdout.json \
        --model_path ./checkpoints/lora_adapter_final \
        --is_lora_adapter
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    records: list[dict],
    model,
    processor,
    processed_root: Path,
    batch_size: int = 8,
    max_slices: int = 16,
    device: str = "cuda",
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (embeddings, volume_ids).
    Each embedding is the mean-pooled last hidden state of the model
    over the image tokens, normalised to unit norm.
    """
    model.eval()
    all_embeddings = []
    all_ids = []

    for i in tqdm(range(0, len(records), batch_size), desc="Extracting embeddings"):
        batch = records[i : i + batch_size]

        for record in batch:
            volume_id = record["volume_id"]

            # Load slice images
            images = []
            for rel_path in record["images"][:max_slices]:
                img_path = processed_root / rel_path
                if img_path.exists():
                    images.append(Image.open(img_path).convert("RGB"))

            if not images:
                continue

            # Use only the retrieval prompt (no target text) for embedding
            query = record["conversations"][0]["value"]

            inputs = processor(
                text=query,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Mean-pool the last hidden state over all non-padding tokens
            hidden = outputs.hidden_states[-1]           # (1, seq_len, hidden_dim)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            embedding = embedding.squeeze(0).cpu().float().numpy()

            # L2 normalise
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            all_embeddings.append(embedding)
            all_ids.append(volume_id)

    return np.stack(all_embeddings, axis=0), all_ids


# ── Recall@K ─────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    query_embeddings: np.ndarray,
    index_embeddings: np.ndarray,
    query_ids: list[str],
    index_ids: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """
    For each query, retrieve top-max(k_values) results from the index
    and compute Recall@K. A hit is when the correct volume appears in
    the top-K (excluding the query itself if it is in the index).
    """
    dim = index_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)   # inner product = cosine on unit vecs
    faiss_index.add(index_embeddings.astype(np.float32))

    max_k = max(k_values) + 1             # +1 to account for self-retrieval
    _, top_indices = faiss_index.search(query_embeddings.astype(np.float32), max_k)

    hits = {k: 0 for k in k_values}

    for q_idx, row in enumerate(top_indices):
        q_id = query_ids[q_idx]
        # Exclude self-match
        retrieved_ids = [
            index_ids[idx] for idx in row if index_ids[idx] != q_id
        ]
        for k in k_values:
            if q_id in retrieved_ids[:k]:
                hits[k] += 1

    n = len(query_ids)
    return {f"Recall@{k}": round(hits[k] / n, 4) for k in k_values}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--holdout_json", default="/data/processed/ctrate_holdout.json")
    p.add_argument("--model_path", default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--is_lora_adapter", action="store_true",
                   help="Set if model_path points to a saved LoRA adapter")
    p.add_argument("--base_model", default="MONAI/Llama3-VILA-M3-8B",
                   help="Base model to load before applying LoRA adapter")
    p.add_argument("--output_json", default="results/retrieval_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_root = Path(cfg["data"]["processed_root"])
    k_values = cfg["eval"]["retrieval_k"]

    # Register VILA-M3 custom architecture with transformers
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # type: ignore  # noqa
    except ImportError:
        pass

    # Load model
    print(f"Loading model: {args.model_path}")
    if args.is_lora_adapter:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model.eval()

    # Load hold-out records (retrieval type only)
    with open(args.holdout_json) as f:
        all_records = json.load(f)

    retrieval_records = [r for r in all_records if r["type"] == "retrieval"]
    print(f"Hold-out retrieval records: {len(retrieval_records)}")

    # Extract embeddings for all hold-out volumes
    embeddings, volume_ids = extract_embeddings(
        retrieval_records,
        model,
        processor,
        processed_root,
        batch_size=cfg["eval"]["batch_size"],
        max_slices=cfg["data"]["max_slices"],
        device=device,
    )

    print(f"Extracted {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    # Compute Recall@K (query = each volume's text, corpus = all embeddings)
    recall = compute_recall_at_k(
        query_embeddings=embeddings,
        index_embeddings=embeddings,
        query_ids=volume_ids,
        index_ids=volume_ids,
        k_values=k_values,
    )

    print("\n── Retrieval Results ──────────────────────")
    for metric, value in recall.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model_path,
        "n_queries": len(volume_ids),
        "metrics": recall,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
