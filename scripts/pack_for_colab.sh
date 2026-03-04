#!/usr/bin/env bash
# ==================================================================
# Pack features & data for Colab upload
# ==================================================================
# Creates per-directory archives of pre-extracted features.
# moss_multi is split into shards (~500 files each, ~5 GB per shard)
# to avoid incomplete archives from macOS timeouts on huge single files.
#
# Usage:
#   bash scripts/pack_for_colab.sh              # pack all 4 dirs
#   bash scripts/pack_for_colab.sh multi_layer  # pack only multi_layer needs
#   bash scripts/pack_for_colab.sh last_layer   # pack only last_layer needs
#   bash scripts/pack_for_colab.sh --force       # re-pack even if archives exist
# ==================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# Parse flags
FORCE=false
MODE="all"
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        *) MODE="$arg" ;;
    esac
done

FEAT_DIR="data/features"
OUT_DIR="archives"
SHARD_SIZE=500  # files per shard for moss_multi
mkdir -p "${OUT_DIR}"

echo "Packing features for Colab (mode: ${MODE}, force: ${FORCE})..."
echo ""

# Count files
for d in clean_dac noisy_dac moss_last moss_multi; do
    n=$(ls "${FEAT_DIR}/$d/"*.pt 2>/dev/null | wc -l | tr -d ' ')
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

# ---- Helper: pack a single directory into one archive ----
pack_single() {
    local d="$1"
    local ARCHIVE="${OUT_DIR}/features_${d}.tar.gz"
    local expected
    expected=$(ls "${FEAT_DIR}/$d/"*.pt 2>/dev/null | wc -l | tr -d ' ')

    # Skip if archive exists, is complete, and --force not set
    if [[ -f "${ARCHIVE}" ]] && [[ "${FORCE}" == "false" ]]; then
        local actual
        actual=$(tar tzf "${ARCHIVE}" 2>/dev/null | grep -c '\.pt$' || echo 0)
        if [[ "${actual}" -eq "${expected}" ]]; then
            echo "  ⏭  ${ARCHIVE} already complete (${actual}/${expected} files), skipping"
            return
        else
            echo "  ⚠️  ${ARCHIVE} incomplete (${actual}/${expected} files), repacking..."
            rm -f "${ARCHIVE}"
        fi
    elif [[ -f "${ARCHIVE}" ]] && [[ "${FORCE}" == "true" ]]; then
        echo "  🔄 --force: removing old ${ARCHIVE}"
        rm -f "${ARCHIVE}"
    fi

    echo "  Archiving ${FEAT_DIR}/${d} → ${ARCHIVE} ..."
    COPYFILE_DISABLE=1 tar czf "${ARCHIVE}" -C "${FEAT_DIR}" "${d}/"
    local actual
    actual=$(tar tzf "${ARCHIVE}" | grep -c '\.pt$' || echo 0)
    local SIZE
    SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
    if [[ "${actual}" -eq "${expected}" ]]; then
        echo "    ✅ ${ARCHIVE} (${SIZE}, ${actual}/${expected} files)"
    else
        echo "    ❌ ${ARCHIVE} INCOMPLETE (${SIZE}, ${actual}/${expected} files)"
        exit 1
    fi
}

# ---- Helper: pack a large directory into sharded archives ----
pack_sharded() {
    local d="$1"
    local all_files
    all_files=($(ls "${FEAT_DIR}/$d/"*.pt | sort))
    local total=${#all_files[@]}
    local num_shards=$(( (total + SHARD_SIZE - 1) / SHARD_SIZE ))

    echo "  Sharded packing: ${d} (${total} files → ${num_shards} shards of ≤${SHARD_SIZE})"

    local shard=0
    local offset=0
    while [[ $offset -lt $total ]]; do
        local end=$(( offset + SHARD_SIZE ))
        if [[ $end -gt $total ]]; then end=$total; fi
        local count=$(( end - offset ))

        local ARCHIVE="${OUT_DIR}/features_${d}_shard${shard}.tar.gz"

        # Skip if complete and not forced
        if [[ -f "${ARCHIVE}" ]] && [[ "${FORCE}" == "false" ]]; then
            local actual
            actual=$(tar tzf "${ARCHIVE}" 2>/dev/null | grep -c '\.pt$' || echo 0)
            if [[ "${actual}" -eq "${count}" ]]; then
                echo "    ⏭  shard ${shard}: ${ARCHIVE} complete (${actual}/${count}), skipping"
                offset=$end
                shard=$(( shard + 1 ))
                continue
            else
                echo "    ⚠️  shard ${shard}: ${ARCHIVE} incomplete (${actual}/${count}), repacking..."
                rm -f "${ARCHIVE}"
            fi
        elif [[ -f "${ARCHIVE}" ]] && [[ "${FORCE}" == "true" ]]; then
            rm -f "${ARCHIVE}"
        fi

        # Create a temp file list with just filenames (relative to FEAT_DIR)
        local FILELIST
        FILELIST=$(mktemp)
        for (( i=offset; i<end; i++ )); do
            echo "${d}/$(basename "${all_files[$i]}")" >> "${FILELIST}"
        done

        echo "    Shard ${shard}: files ${offset}..$(( end - 1 )) (${count} files) ..."
        COPYFILE_DISABLE=1 tar czf "${ARCHIVE}" -C "${FEAT_DIR}" -T "${FILELIST}"
        rm -f "${FILELIST}"

        local actual
        actual=$(tar tzf "${ARCHIVE}" | grep -c '\.pt$' || echo 0)
        local SIZE
        SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
        if [[ "${actual}" -eq "${count}" ]]; then
            echo "    ✅ shard ${shard}: ${ARCHIVE} (${SIZE}, ${actual} files)"
        else
            echo "    ❌ shard ${shard}: INCOMPLETE (${actual}/${count})"
            exit 1
        fi

        offset=$end
        shard=$(( shard + 1 ))
    done
}

# ---- Main packing loop ----
for d in ${DIRS}; do
    if [[ "$d" == "moss_multi" ]]; then
        # Remove old single-file archive if it exists (it was likely incomplete)
        if [[ -f "${OUT_DIR}/features_moss_multi.tar.gz" ]]; then
            echo "  🗑  Removing old single-file archive: features_moss_multi.tar.gz"
            rm -f "${OUT_DIR}/features_moss_multi.tar.gz"
        fi
        pack_sharded "$d"
    else
        pack_single "$d"
    fi
    echo ""
done

echo "=============================================="
echo "  Archives saved to ${OUT_DIR}/"
ls -lh "${OUT_DIR}"/features_*.tar.gz 2>/dev/null
echo "=============================================="
echo ""
echo "Upload the archives/ folder to Google Drive, then in Colab the"
echo "notebook will automatically unpack all archives (including shards)."
