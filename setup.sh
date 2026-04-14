#!/usr/bin/env bash
# setup.sh — run once on the server before anything else
# Usage: bash setup.sh
set -e

VILA_M3_REPO="$HOME/projects/VILA-M3_nafis/VLM-Radiology-Agent-Framework"
VENV_DIR="$HOME/projects/icsdg_venv"

echo "==> Creating virtual environment at $VENV_DIR"
python3.10 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Installing VILA-M3 in editable mode"
pip install -e "$VILA_M3_REPO"

echo "==> Installing flash-attention (may take a few minutes)"
# Required by VILA-M3 for efficient attention on long sequences
pip install flash-attn --no-build-isolation

echo "==> Creating data directories"
mkdir -p /data/ct_rate /data/lidc_idri /data/processed

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
