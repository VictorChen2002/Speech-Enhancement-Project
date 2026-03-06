#!/usr/bin/env bash
# ==================================================================
# Pack Multi-SNR Features for Colab Upload
# ==================================================================
# Archives the combined multi-SNR feature directory for uploading
# to Google Drive. moss_multi is auto-sharded (~500 files each).
#
# Upload destination: MyDrive/speech_enhancement_features_multi_snr/
#
# Usage:
#   bash scripts/pack_multi_snr.sh            # pack all
#   bash scripts/pack_multi_snr.sh --no-multi # skip moss_multi (large)
# ==================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

FEAT_DIR="data/features_multi_snr"
OUT_DIR="archives/multi_snr"
SHARD_SIZE=500
SKIP_MULTI=false

for arg in "$@"; do
    [[ "$arg" == "--no-multi" ]] && SKIP_MULTI=true
done

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo " Pack Multi-SNR Features"
echo " Source: ${FEAT_DIR}/"
echo " Output: ${OUT_DIR}/"
echo "=============================================="
echo ""

# Show stats
for d in clean_dac noisy_dac moss_last moss_multi; do
    dir="${FEAT_DIR}/$d"
    if [[ -d "$dir" ]]; then
        n=$(ls "$dir"/*.pt 2>/dev/null | wc -l | tr -d ' ')
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $d: $n files ($sz)"
    else
        echo "  $d: NOT FOUND"
    fi
done
echo ""

# --- Pack clean_dac, noisy_dac, moss_last ---
for d in clean_dac noisy_dac moss_last; do
    archive="${OUT_DIR}/features_${d}.tar.gz"
    dir="${FEAT_DIR}/$d"
    if [[ ! -d "$dir" ]]; then
        echo "[SKIP] ${d}/ not found"
        continue
    fi
    if [[ -f "$archive" ]]; then
        echo "[OK]   ${archive} already exists"
        continue
    fi
    echo "Packing ${d}/ ..."
    tar czf "$archive" -C "$dir" .
    echo "  -> $(du -sh "$archive" | cut -f1)"
done

# --- Pack moss_multi in shards ---
if $SKIP_MULTI; then
    echo ""
    echo "[SKIP] moss_multi (--no-multi flag)"
else
    MULTI_DIR="${FEAT_DIR}/moss_multi"
    if [[ ! -d "$MULTI_DIR" ]]; then
        echo "[SKIP] moss_multi/ not found"
    else
        echo ""
        echo "Sharding moss_multi/ (${SHARD_SIZE} files per shard)..."
        files=("${MULTI_DIR}"/*.pt)
        total=${#files[@]}
        shard=1

        for ((i=0; i<total; i+=SHARD_SIZE)); do
            shard_name=$(printf "features_moss_multi_shard%02d.tar.gz" "$shard")
            archive="${OUT_DIR}/${shard_name}"
            if [[ -f "$archive" ]]; then
                echo "  [OK] ${shard_name} already exists"
                ((shard++))
                continue
            fi
            end=$((i + SHARD_SIZE))
            [[ $end -gt $total ]] && end=$total
            slice=("${files[@]:$i:$((end - i))}")
            echo "  Shard ${shard}: files $((i+1))–${end} of ${total}"

            # Create temp file list
            tmp_list=$(mktemp)
            for f in "${slice[@]}"; do
                basename "$f" >> "$tmp_list"
            done
            tar czf "$archive" -C "$MULTI_DIR" -T "$tmp_list"
            rm "$tmp_list"
            echo "    -> $(du -sh "$archive" | cut -f1)"
            ((shard++))
        done
    fi
fi

echo ""
echo "=============================================="
echo " Archives ready in ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
echo ""
echo " Upload to: MyDrive/speech_enhancement_features_multi_snr/"
echo "=============================================="
