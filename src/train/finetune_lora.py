"""
finetune_lora.py
----------------
LoRA fine-tuning of VILA-M3 8B on CT-RATE instruction pairs.

This script wraps the VILA-M3 training infrastructure with a PEFT LoRA adapter.
It supports both detection and retrieval instruction types from prepare_ctrate.py.

Usage:
    python src/train/finetune_lora.py \
        --config configs/train_config.yaml \
        --data_path /data/processed/ctrate_train.json \
        --vila_repo ../VLM-Radiology-Agent-Framework

For multi-GPU training (all 3 L40s):
    torchrun --nproc_per_node=3 src/train/finetune_lora.py \
        --config configs/train_config.yaml \
        --data_path /data/processed/ctrate_train.json \
        --vila_repo ../VLM-Radiology-Agent-Framework
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── Dataset ───────────────────────────────────────────────────────────────────

class CTRATEInstructionDataset(Dataset):
    """
    Loads instruction-following records from ctrate_train.json and prepares
    model inputs. Each record contains a list of slice image paths and a
    two-turn conversation (human prompt → gpt response).
    """

    def __init__(
        self,
        data_path: str,
        processor,
        processed_root: str,
        max_slices: int = 16,
        max_length: int = 2048,
    ):
        with open(data_path) as f:
            self.records = json.load(f)

        self.processor = processor
        self.processed_root = Path(processed_root)
        self.max_slices = max_slices
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        record = self.records[idx]

        # Load slice images
        images = []
        for rel_path in record["images"][: self.max_slices]:
            img_path = self.processed_root / rel_path
            if img_path.exists():
                images.append(Image.open(img_path).convert("RGB"))

        # Build conversation text
        human_turn = record["conversations"][0]["value"]
        gpt_turn = record["conversations"][1]["value"]
        full_text = f"USER: {human_turn}\nASSISTANT: {gpt_turn}"

        # Tokenise via the VILA-M3 processor
        if images:
            inputs = self.processor(
                text=full_text,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self.processor(
                text=full_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        # Flatten the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Labels: mask the human turn so the loss is only on the gpt response
        labels = inputs["input_ids"].clone()
        # Find where the ASSISTANT token starts and mask everything before it
        assistant_token_id = self.processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )[0]
        assistant_pos = (labels == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_pos) > 0:
            labels[: assistant_pos[0] + 1] = -100  # -100 = ignore in loss

        inputs["labels"] = labels
        return inputs


# ── LoRA setup ────────────────────────────────────────────────────────────────

def build_lora_model(model_name: str, lora_cfg: dict):
    """Load VILA-M3 and attach a LoRA adapter."""
    print(f"Loading base model: {model_name}")

    # Register VILA-M3 custom architecture with transformers
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # type: ignore  # noqa
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--data_path", default="/data/processed/ctrate_train.json")
    p.add_argument("--vila_repo", default=None,
                   help="Path to VLM-Radiology-Agent-Framework repo root (added to sys.path)")
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint directory to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    # Optionally add the VILA-M3 repo to the Python path so its modules are importable
    if args.vila_repo:
        vila_repo = str(Path(args.vila_repo).expanduser())
        if vila_repo not in sys.path:
            sys.path.insert(0, vila_repo)
            sys.path.insert(0, str(Path(vila_repo) / "m3"))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])

    model_name = cfg["model"]["name"]

    # Load processor (handles both tokenisation and image preprocessing)
    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Build LoRA model
    model = build_lora_model(model_name, cfg["lora"])

    # Dataset
    dataset = CTRATEInstructionDataset(
        data_path=args.data_path,
        processor=processor,
        processed_root=cfg["data"]["processed_root"],
        max_slices=cfg["data"]["max_slices"],
    )
    print(f"Training samples: {len(dataset)}")

    # TrainingArguments
    train_cfg = cfg["train"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        save_strategy=train_cfg["save_strategy"],
        logging_steps=train_cfg["logging_steps"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="none",           # disable W&B / wandb by default
        seed=train_cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training ...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save the final LoRA adapter (much smaller than the full model)
    adapter_path = Path(train_cfg["output_dir"]) / "lora_adapter_final"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
