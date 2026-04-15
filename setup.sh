#!/usr/bin/env bash
# setup.sh — run once on the server before anything else
# Usage: bash setup.sh
#
# Assumes conda is available (base environment active).
# Creates a conda environment named 'icsdg' with Python 3.10.
set -e

CONDA_ENV="icsdg"
VILA_M3_REPO="$HOME/projects/VILA-M3_nafis/VLM-Radiology-Agent-Framework"

echo "==> Creating conda environment: $CONDA_ENV (python=3.10)"
conda create -y -n "$CONDA_ENV" python=3.10

echo "==> Activating $CONDA_ENV"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Installing VILA-M3 in editable mode"
pip install -e "$VILA_M3_REPO"

echo "==> Installing flash-attention (may take a few minutes)"
pip install flash-attn --no-build-isolation

echo "==> Creating data directories"
mkdir -p /data/ct_rate /data/lidc_idri /data/processed

echo ""
echo "Setup complete. Activate with:"
echo "  conda activate $CONDA_ENV"
