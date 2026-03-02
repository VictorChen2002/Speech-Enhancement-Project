#!/usr/bin/env bash
# ==================================================================
# Overnight Data-Preparation Pipeline  (v2 — robust)
# ==================================================================
# Runs: mixer (multiple SNRs) → extract_dac (clean + noisy) →
#        extract_moss (last_layer + multi_layer for noisy)
#
# Improvements over v1:
#   • Each step is guarded — a failure logs the error but does NOT
#     abort later steps (no set -e for the main body).
#   • Timestamped log output for easier debugging.
#   • Skip steps whose output directories already contain data
#     (use --force to re-run everything).
#   • Designed for long-running background execution:
#       nohup bash scripts/run_overnight_v2.sh > overnight.log 2>&1 &
# ==================================================================

cd "$(dirname "$0")/.."   # project root

# --- file-descriptor hygiene (critical for nohup) --------------------------
# nohup closes stdin; Python crashes on startup if fd 0 is invalid.
# Redirect stdin from /dev/null and make sure stdout/stderr are real files.
exec 0</dev/null
# If stdout is not a terminal and not a file (broken fd), reopen to /dev/null
if ! { true >&1; } 2>/dev/null; then exec 1>/dev/null; fi
if ! { true >&2; } 2>/dev/null; then exec 2>&1; fi
# ---------------------------------------------------------------------------

# --- configurable -------------------------------------------------------
TRAIN_SNR="${1:-5}"                          # SNR (dB) used for feature extraction
FORCE="${2:-}"                               # pass "force" as 2nd arg to re-run
SNR_LIST="-5 0 5 10 15"                      # all SNR levels for mixing
CLEAN_DIR="data/raw/clean"
NOISE_DIR="data/raw/noise"
MIX_DIR="data/mixed"
FEAT_DIR="data/features"
SR=16000
DEVICE="auto"                                # auto-detects cuda > mps > cpu
# -------------------------------------------------------------------------

ts() { date "+%Y-%m-%d %H:%M:%S"; }
SUCCESS=0
FAIL=0
SKIP=0

echo "=============================================="
echo " Overnight Pipeline v2  (train SNR = ${TRAIN_SNR} dB)"
echo " Started at $(ts)"
echo "=============================================="

# Helper: run a step, catch errors
run_step() {
    local step_name="$1"; shift
    local out_check_dir="$1"; shift  # directory to check for existing output
    local desc="$1"; shift

    echo ""
    echo "----------------------------------------------"
    echo "[$(ts)] ${step_name}: ${desc}"
    echo "----------------------------------------------"

    # Skip if output already has all expected files (and --force not set)
    # For extraction steps, the python scripts have built-in resume (skip
    # existing .pt files), so we only skip at the bash level if the
    # directory already has the full expected count (EXPECTED_COUNT).
    if [[ -z "$FORCE" && -n "$out_check_dir" && -d "$out_check_dir" ]]; then
        local actual
        actual=$(find "$out_check_dir" -type f -name '*.pt' -o -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$actual" -ge "${EXPECTED_COUNT:-1}" ]]; then
            echo "  ⏭  Skipping — ${out_check_dir} already has ${actual} files (expected ${EXPECTED_COUNT:-1})"
            echo "     (pass 'force' as 2nd arg to re-run)"
            SKIP=$((SKIP + 1))
            return 0
        elif [[ "$actual" -gt 0 ]]; then
            echo "  ↪  Resuming — ${out_check_dir} has ${actual}/${EXPECTED_COUNT:-'?'} files"
        fi
    fi

    # Run the command (stdin from /dev/null to prevent fd issues under nohup)
    if "$@" </dev/null; then
        echo "[$(ts)] ✅  ${step_name} completed successfully."
        SUCCESS=$((SUCCESS + 1))
    else
        local rc=$?
        echo "[$(ts)] ❌  ${step_name} FAILED (exit code ${rc})."
        FAIL=$((FAIL + 1))
    fi
}

# Count expected files from clean audio directory
EXPECTED_COUNT=$(find "${CLEAN_DIR}" -type f \( -name '*.wav' -o -name '*.flac' -o -name '*.mp3' \) 2>/dev/null | wc -l | tr -d ' ')
echo "Expected file count per feature dir: ${EXPECTED_COUNT}"
export EXPECTED_COUNT

# ======================= Step 1 — Mixer ==================================
run_step "Step 1/5" "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
    "Mixing clean + noise at SNRs: ${SNR_LIST}" \
    python -m src.data.mixer \
        --clean_dir "${CLEAN_DIR}" \
        --noise_dir "${NOISE_DIR}" \
        --out_dir   "${MIX_DIR}" \
        --snr_list  ${SNR_LIST} \
        --sr        ${SR}

# ======================= Step 2 — DAC clean ==============================
run_step "Step 2/5" "${FEAT_DIR}/clean_dac" \
    "Extracting DAC latents from clean audio" \
    python -m src.data.extract_dac \
        --audio_dir "${CLEAN_DIR}" \
        --out_dir   "${FEAT_DIR}/clean_dac" \
        --sr        ${SR} \
        --device    ${DEVICE}

# ======================= Step 3 — DAC noisy ==============================
run_step "Step 3/5" "${FEAT_DIR}/noisy_dac" \
    "Extracting DAC latents from noisy audio (SNR=${TRAIN_SNR}dB)" \
    python -m src.data.extract_dac \
        --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
        --out_dir   "${FEAT_DIR}/noisy_dac" \
        --sr        ${SR} \
        --device    ${DEVICE}

# ======================= Step 4 — MOSS last-layer ========================
run_step "Step 4/5" "${FEAT_DIR}/moss_last" \
    "Extracting MOSS last-layer embeddings (noisy, SNR=${TRAIN_SNR}dB)" \
    python -m src.data.extract_moss \
        --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
        --out_dir   "${FEAT_DIR}/moss_last" \
        --sr        ${SR} \
        --device    ${DEVICE}

# ======================= Step 5 — MOSS multi-layer =======================
run_step "Step 5/5" "${FEAT_DIR}/moss_multi" \
    "Extracting MOSS multi-layer embeddings (noisy, SNR=${TRAIN_SNR}dB)" \
    python -m src.data.extract_moss \
        --audio_dir "${MIX_DIR}/snr_${TRAIN_SNR}dB" \
        --out_dir   "${FEAT_DIR}/moss_multi" \
        --sr        ${SR} \
        --device    ${DEVICE} \
        --save_all_layers

# ======================= Summary ========================================
echo ""
echo "=============================================="
echo " Pipeline finished at $(ts)"
echo " Results:  ✅ ${SUCCESS} succeeded | ❌ ${FAIL} failed | ⏭ ${SKIP} skipped"
echo "=============================================="
echo ""
echo "Expected layout:"
echo "  ${FEAT_DIR}/clean_dac/*.pt    (T_dac, 1024)  50 Hz"
echo "  ${FEAT_DIR}/noisy_dac/*.pt    (T_dac, 1024)  50 Hz"
echo "  ${FEAT_DIR}/moss_last/*.pt    (T_moss, 768)  12.5 Hz"
echo "  ${FEAT_DIR}/moss_multi/*.pt   list[(T_moss, 1280)] × 32 layers"
echo ""
if [[ $FAIL -gt 0 ]]; then
    echo "⚠️  Some steps failed — check the log above for details."
    exit 1
fi
echo "Next: python train.py --config configs/default.yaml --condition_type multi_layer"
