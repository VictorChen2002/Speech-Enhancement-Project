#!/usr/bin/env bash
# ==================================================================
# Pack features & data for Colab upload
# ==================================================================
# Creates per-directory archives of pre-extracted features.
# Each dir is archived separately so large dirs (moss_multi ~17GB)
# don't cause macOS APFS sparse-file timeouts.
#
# Usage:
#   bash scripts/pack_for_colab.sh              # pack all 4 dirs
#   bash scripts/pack_for_colab.sh multi_layer  # pack only multi_layer needs
#   bash scripts/pack_for_colab.sh last_layer   # pack only last_layer needs
# ==================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"
FEAT_DIR="data/features"
OUT_DIR="archives"
mkdir -p "${OUT_DIR}"

echo "Packing features for Colab (mode: ${MODE})..."
echo ""

# Count files
for d in clean_dac noisy_dac moss_last moss_multi; do
    n=$(ls "${FEAT_DIR}/$d/" 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "${FEAT_DIR}/$d" 2>/dev/null | cut -f1)
    echo "  ${FEAT_DIR}/$d: $n files ($sz)"
done
echo ""

# Decide which dirs to pack
case "${MODE}" in
    multi_layer)
        DIRS="clean_dac noisy_dac moss_multi"
        echo "Packing for multi_layer training (clean_dac + noisy_dac + moss_multi)"
        ;;
    last_layer)
        DIRS="clean_dac noisy_dac moss_last"
        echo "Packing for last_layer training (clean_dac + noisy_dac + moss_last)"
        ;;
    none)
        DIRS="clean_dac noisy_dac"
        echo "Packing for unconditioned training (clean_dac + noisy_dac)"
        ;;
    all|*)
        DIRS="clean_dac noisy_dac moss_last moss_multi"
        echo "Packing all feature directories"
        ;;
esac
echo ""

# Archive each directory separately (avoids macOS APFS sparse-file timeouts)
for d in ${DIRS}; do
    ARCHIVE="${OUT_DIR}/features_${d}.tar.gz"
    if [[ -f "${ARCHIVE}" ]]; then
        echo "  ⏭  ${ARCHIVE} already exists, skipping"
        continue
    fi
    echo "  Archiving ${FEAT_DIR}/${d} → ${ARCHIVE} ..."
    # COPYFILE_DISABLE=1 prevents macOS ._* resource fork files
    COPYFILE_DISABLE=1 tar czf "${ARCHIVE}" -C "${FEAT_DIR}" "${d}/"
    SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
    echo "    ✅ ${ARCHIVE} (${SIZE})"
done

echo ""
echo "=============================================="
echo "  Archives saved to ${OUT_DIR}/"
ls -lh "${OUT_DIR}"/features_*.tar.gz 2>/dev/null
echo "=============================================="
echo ""
echo "Upload the archives/ folder to Google Drive, then in Colab run:"
echo ""
echo '  !for f in /content/drive/MyDrive/archives/features_*.tar.gz; do'
echo '      tar xzf "$f" -C /content/speech-enhancement-project/data/features/; done'
