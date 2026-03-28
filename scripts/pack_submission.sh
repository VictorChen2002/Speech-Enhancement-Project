#!/usr/bin/env bash
# pack_submission.sh — Create a clean code-only submission archive.
#
# Usage:
#     bash scripts/pack_submission.sh
#
# Output:
#     submission_package/             (folder, ready to zip)
#     submission_package.zip          (final archive)

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$SRC_DIR/submission_package"

echo "=== Packing submission from: $SRC_DIR ==="

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# ── Core source code ─────────────────────────────────────────────────────
cp "$SRC_DIR/train.py"      "$OUT_DIR/"
cp "$SRC_DIR/evaluate.py"   "$OUT_DIR/"
cp "$SRC_DIR/demo.py"       "$OUT_DIR/"
cp "$SRC_DIR/requirements.txt" "$OUT_DIR/"

# Configs
mkdir -p "$OUT_DIR/configs"
cp "$SRC_DIR/configs/default.yaml"     "$OUT_DIR/configs/"
cp "$SRC_DIR/configs/small_model.yaml" "$OUT_DIR/configs/" 2>/dev/null || true

# Source modules
mkdir -p "$OUT_DIR/src/data" "$OUT_DIR/src/models" "$OUT_DIR/src/utils"
cp "$SRC_DIR/src/__init__.py"              "$OUT_DIR/src/"
cp "$SRC_DIR/src/data/mixer.py"            "$OUT_DIR/src/data/"
cp "$SRC_DIR/src/data/extract_dac.py"      "$OUT_DIR/src/data/"
cp "$SRC_DIR/src/data/extract_moss.py"     "$OUT_DIR/src/data/"
touch "$OUT_DIR/src/data/__init__.py"
cp "$SRC_DIR/src/models/dit.py"            "$OUT_DIR/src/models/"
cp "$SRC_DIR/src/models/flow_matching.py"  "$OUT_DIR/src/models/"
touch "$OUT_DIR/src/models/__init__.py"
cp "$SRC_DIR/src/utils/metrics.py"         "$OUT_DIR/src/utils/"
cp "$SRC_DIR/src/utils/viz.py"             "$OUT_DIR/src/utils/"
touch "$OUT_DIR/src/utils/__init__.py"

# Shell scripts
mkdir -p "$OUT_DIR/scripts"
for f in "$SRC_DIR"/scripts/*.sh; do
    [ -f "$f" ] && cp "$f" "$OUT_DIR/scripts/"
done

# Notebooks (analysis only — skip colab infra notebooks)
mkdir -p "$OUT_DIR/notebooks"
cp "$SRC_DIR/notebooks/analysis.ipynb"     "$OUT_DIR/notebooks/" 2>/dev/null || true
cp "$SRC_DIR/notebooks/inspection.ipynb"   "$OUT_DIR/notebooks/" 2>/dev/null || true

echo "=== Files packed ==="
find "$OUT_DIR" -type f | sort | sed "s|$OUT_DIR/||"

# ── Create zip ───────────────────────────────────────────────────────────
cd "$SRC_DIR"
rm -f submission_package.zip
zip -r submission_package.zip submission_package/ -x '*.DS_Store' '*__pycache__*'
echo ""
echo "=== Done: submission_package.zip ==="
du -sh submission_package.zip
