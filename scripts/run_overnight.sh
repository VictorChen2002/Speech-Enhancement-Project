#!/usr/bin/env bash
# ==================================================================
# Overnight Data-Preparation Pipeline
# ==================================================================
# Runs: mixer (multiple SNRs) → extract_dac (clean + noisy) →
#        extract_moss (last_layer + multi_layer for noisy)
#
# Usage:
#   conda activate torch
#   bash scripts/run_overnight.sh          # default SNR=5dB for features
#   bash scripts/run_overnight.sh 0        # use SNR=0dB for features
#   nohup bash scripts/run_overnight.sh > overnight.log 2>&1 &
# ==================================================================
set -euo pipefail

cd "$(dirname "$0")/.."  # project root

# --- configurable ---
TRAIN_SNR="${1:-5}"                          # SNR (dB) used for feature extraction
SNR_LIST="-5 0 5 10 15"                      # all SNR levels for mixing
CLEAN_DIR="data/raw/clean"
NOISE_DIR="data/raw/noise"
MIX_DIR="data/mixed"
FEAT_DIR="data/features"
SR=16000
DEVICE="auto"                                # auto-detects cuda > mps > cpu

echo "=============================================="
echo " Overnight Pipeline  (train SNR = ${TRAIN_SNR} dB)"
echo "=============================================="

# Step 1 — Mixer: create noisy audio at multiple SNR levels
echo ""
echo "[1/5] Mixing clean + noise at SNRs: ${SNR_LIST}"
python -m src.data.mixer \
    --clean_dir "${CLEAN_DIR}" \
    --noise_dir "${NOISE_DIR}" \
    --out_dir   "${MIX_DIR}" \
    --snr_list  ${SNR_LIST} \
    --sr        ${SR}

# Step 2 — Extract DAC latents from CLEAN audio
echo ""
echo "[2/5] Extracting DAC latents from clean audio"
python -m src.data.extract_dac \
    --audio_dir "${CLEAN_DIR}" \
    --out_dir   "${FEAT_DIR}/clean_dac" \
    --sr        ${SR} \
    --device    ${DEVICE}

# Step 3 — Extract DAC latents from NOISY audio (training SNR)
echo ""
echo "[3/5] Extracting DAC latents from noisy audio (SNR=${TRAIN_SNR}dB)"
python -m src.data.extract_dac \
    --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
    --out_dir   "${FEAT_DIR}/noisy_dac" \
    --sr        ${SR} \
    --device    ${DEVICE}

# Step 4 — Extract MOSS last-layer embeddings from NOISY audio
echo ""
echo "[4/5] Extracting MOSS last-layer embeddings (noisy, SNR=${TRAIN_SNR}dB)"
python -m src.data.extract_moss \
    --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
    --out_dir   "${FEAT_DIR}/moss_last" \
    --sr        ${SR} \
    --device    ${DEVICE}

# Step 5 — Extract MOSS multi-layer embeddings from NOISY audio
echo ""
echo "[5/5] Extracting MOSS multi-layer embeddings (noisy, SNR=${TRAIN_SNR}dB)"
python -m src.data.extract_moss \
    --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
    --out_dir   "${FEAT_DIR}/moss_multi" \
    --sr        ${SR} \
    --device    ${DEVICE} \
    --save_all_layers

echo ""
echo "=============================================="
echo " All done!  Features stored in ${FEAT_DIR}/"
echo "=============================================="
echo ""
echo "Expected layout:"
echo "  ${FEAT_DIR}/clean_dac/*.pt    (T_dac, 1024)  50 Hz"
echo "  ${FEAT_DIR}/noisy_dac/*.pt    (T_dac, 1024)  50 Hz"
echo "  ${FEAT_DIR}/moss_last/*.pt    (T_moss, 768)  12.5 Hz"
echo "  ${FEAT_DIR}/moss_multi/*.pt   list[(T_moss, 1280)] × 32 layers"
echo ""
echo "Next: python train.py --config configs/default.yaml --condition_type multi_layer"
