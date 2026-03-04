#!/usr/bin/env bash
# Quick script: shard-pack only moss_multi (skip other archives)
set -euo pipefail
cd "$(dirname "$0")/.."

FEAT_DIR="data/features"
OUT_DIR="archives"
SHARD_SIZE=500
SRC="${FEAT_DIR}/moss_multi"

# Remove any old single-file archive
rm -f "${OUT_DIR}/features_moss_multi.tar.gz"

# Collect all .pt files sorted
ALL_FILES=()
while IFS= read -r f; do
    ALL_FILES+=("$f")
done < <(ls "${SRC}/"*.pt | sort)
TOTAL=${#ALL_FILES[@]}
NUM_SHARDS=$(( (TOTAL + SHARD_SIZE - 1) / SHARD_SIZE ))

echo "Packing moss_multi: ${TOTAL} files → ${NUM_SHARDS} shards of ≤${SHARD_SIZE}"
echo ""

SHARD=0
OFFSET=0
while [[ $OFFSET -lt $TOTAL ]]; do
    END=$(( OFFSET + SHARD_SIZE ))
    if [[ $END -gt $TOTAL ]]; then END=$TOTAL; fi
    COUNT=$(( END - OFFSET ))

    ARCHIVE="${OUT_DIR}/features_moss_multi_shard${SHARD}.tar.gz"

    if [[ -f "${ARCHIVE}" ]]; then
        echo "  Shard ${SHARD}: ${ARCHIVE} exists, skipping (delete to repack)"
        OFFSET=$END
        SHARD=$(( SHARD + 1 ))
        continue
    fi

    # Build file list
    FILELIST=$(mktemp)
    for (( i=OFFSET; i<END; i++ )); do
        echo "moss_multi/$(basename "${ALL_FILES[$i]}")" >> "${FILELIST}"
    done

    echo "  Shard ${SHARD}: files ${OFFSET}..$(( END - 1 )) (${COUNT} files) ..."
    COPYFILE_DISABLE=1 tar czf "${ARCHIVE}" -C "${FEAT_DIR}" -T "${FILELIST}"
    rm -f "${FILELIST}"

    SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
    echo "    ✅ ${ARCHIVE} (${SIZE})"

    OFFSET=$END
    SHARD=$(( SHARD + 1 ))
done

echo ""
echo "Done! Shard archives:"
ls -lh "${OUT_DIR}"/features_moss_multi_shard*.tar.gz
echo ""
echo "Total moss_multi archive size:"
du -sh "${OUT_DIR}"/features_moss_multi_shard*.tar.gz | tail -1
