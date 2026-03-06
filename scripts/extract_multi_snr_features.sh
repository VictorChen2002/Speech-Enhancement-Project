#!/usr/bin/env bash
# ==================================================================
# Multi-SNR Feature Extraction Pipeline
# ==================================================================
# Extracts DAC + MOSS features for multiple SNR levels and combines
# them into a unified feature directory with SNR-tagged filenames.
#
# The unified directory has the same structure as data/features/
# (clean_dac, noisy_dac, moss_last, moss_multi), so the existing
# OfflineFeatureDataset in train.py works without modification.
#
# Usage:
#   bash scripts/extract_multi_snr_features.sh              # all 5 SNRs
#   bash scripts/extract_multi_snr_features.sh "-5 0 5"     # subset
#   nohup bash scripts/extract_multi_snr_features.sh > multi_snr.log 2>&1 &
# ==================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# --- configurable ---
SNR_LIST="${1:--5 0 5 10 15}"
CLEAN_DIR="data/raw/clean"
MIX_DIR="data/mixed"
OUT_DIR="data/features_multi_snr"
TEMP_DIR="data/_temp_snr_extract"
SR=16000
DEVICE="auto"

echo "=============================================="
echo " Multi-SNR Feature Extraction"
echo " SNR levels: ${SNR_LIST}"
echo " Output:     ${OUT_DIR}/"
echo "=============================================="

# ------------------------------------------------------------------
# Step 1: Extract clean DAC (shared across all SNRs, done once)
# ------------------------------------------------------------------
CLEAN_TEMP="${TEMP_DIR}/clean_dac_base"
if [[ -d "${CLEAN_TEMP}" ]] && [[ $(ls "${CLEAN_TEMP}"/*.pt 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[1/3] Clean DAC already extracted, skipping"
else
    echo "[1/3] Extracting clean DAC latents"
    python -m src.data.extract_dac \
        --audio_dir "${CLEAN_DIR}" \
        --out_dir   "${CLEAN_TEMP}" \
        --sr ${SR} --device ${DEVICE}
fi

# ------------------------------------------------------------------
# Step 2: For each SNR, extract noisy DAC + MOSS embeddings
# ------------------------------------------------------------------
echo ""
echo "[2/3] Extracting per-SNR features"

for SNR in ${SNR_LIST}; do
    echo ""
    echo "────────── SNR = ${SNR} dB ──────────"

    NOISY_DIR="${MIX_DIR}/snr_${SNR}dB"
    if [[ ! -d "${NOISY_DIR}" ]]; then
        echo "  [SKIP] ${NOISY_DIR} not found"
        continue
    fi

    # (a) Noisy DAC
    echo "  [a] Extracting noisy DAC"
    python -m src.data.extract_dac \
        --audio_dir "${NOISY_DIR}" \
        --out_dir   "${TEMP_DIR}/noisy_dac_snr${SNR}" \
        --sr ${SR} --device ${DEVICE}

    # (b) MOSS last-layer
    echo "  [b] Extracting MOSS last-layer"
    python -m src.data.extract_moss \
        --audio_dir "${NOISY_DIR}" \
        --out_dir   "${TEMP_DIR}/moss_last_snr${SNR}" \
        --sr ${SR} --device ${DEVICE}

    # (c) MOSS multi-layer
    echo "  [c] Extracting MOSS multi-layer"
    python -m src.data.extract_moss \
        --audio_dir "${NOISY_DIR}" \
        --out_dir   "${TEMP_DIR}/moss_multi_snr${SNR}" \
        --sr ${SR} --device ${DEVICE} \
        --save_all_layers
done

# ------------------------------------------------------------------
# Step 3: Combine into unified directory with SNR-tagged filenames
# ------------------------------------------------------------------
echo ""
echo "[3/3] Combining into unified multi-SNR directory: ${OUT_DIR}/"
mkdir -p "${OUT_DIR}"/{clean_dac,noisy_dac,moss_last,moss_multi}

for SNR in ${SNR_LIST}; do
    SUFFIX="__snr${SNR}"
    echo "  Linking SNR=${SNR}dB files (suffix=${SUFFIX})"

    # Clean DAC — same content per SNR, different filename
    if [[ -d "${CLEAN_TEMP}" ]]; then
        for f in "${CLEAN_TEMP}"/*.pt; do
            [[ -f "$f" ]] || continue
            base=$(basename "$f" .pt)
            ln -f "$f" "${OUT_DIR}/clean_dac/${base}${SUFFIX}.pt" 2>/dev/null \
                || cp "$f" "${OUT_DIR}/clean_dac/${base}${SUFFIX}.pt"
        done
    fi

    # Noisy DAC
    SRC="${TEMP_DIR}/noisy_dac_snr${SNR}"
    if [[ -d "${SRC}" ]]; then
        for f in "${SRC}"/*.pt; do
            [[ -f "$f" ]] || continue
            base=$(basename "$f" .pt)
            ln -f "$f" "${OUT_DIR}/noisy_dac/${base}${SUFFIX}.pt" 2>/dev/null \
                || cp "$f" "${OUT_DIR}/noisy_dac/${base}${SUFFIX}.pt"
        done
    fi

    # MOSS last-layer
    SRC="${TEMP_DIR}/moss_last_snr${SNR}"
    if [[ -d "${SRC}" ]]; then
        for f in "${SRC}"/*.pt; do
            [[ -f "$f" ]] || continue
            base=$(basename "$f" .pt)
            ln -f "$f" "${OUT_DIR}/moss_last/${base}${SUFFIX}.pt" 2>/dev/null \
                || cp "$f" "${OUT_DIR}/moss_last/${base}${SUFFIX}.pt"
        done
    fi

    # MOSS multi-layer
    SRC="${TEMP_DIR}/moss_multi_snr${SNR}"
    if [[ -d "${SRC}" ]]; then
        for f in "${SRC}"/*.pt; do
            [[ -f "$f" ]] || continue
            base=$(basename "$f" .pt)
            ln -f "$f" "${OUT_DIR}/moss_multi/${base}${SUFFIX}.pt" 2>/dev/null \
                || cp "$f" "${OUT_DIR}/moss_multi/${base}${SUFFIX}.pt"
        done
    fi
done

# Summary
TOTAL=$(ls "${OUT_DIR}/clean_dac/"*.pt 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "=============================================="
echo " Done!  ${TOTAL} samples in ${OUT_DIR}/"
echo "   clean_dac : $(ls "${OUT_DIR}/clean_dac/"*.pt 2>/dev/null | wc -l | tr -d ' ') files"
echo "   noisy_dac : $(ls "${OUT_DIR}/noisy_dac/"*.pt 2>/dev/null | wc -l | tr -d ' ') files"
echo "   moss_last : $(ls "${OUT_DIR}/moss_last/"*.pt 2>/dev/null | wc -l | tr -d ' ') files"
echo "   moss_multi: $(ls "${OUT_DIR}/moss_multi/"*.pt 2>/dev/null | wc -l | tr -d ' ') files"
echo "=============================================="
