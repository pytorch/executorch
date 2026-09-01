#!/usr/bin/env bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Orchestrate the four-stage QAT pipeline split example.
#
# Each stage runs in its own Python process and communicates only through files
# in WORKDIR.  This mirrors a real workflow where training and lowering happen
# on separate machines or at different times.
#
# Usage:
#   ./run.sh [--example {minimal|sliced}] [--workdir PATH]
#
# Options:
#   --example   Which of the two example paths to run (default: minimal).
#               minimal : capture with a plain torch.export one-liner, skip
#                         the recipe entirely before QAT, then lower with the
#                         full recipe afterwards.
#               sliced  : slice the recipe around the QUANTIZE stage so that
#                         only the pre-quantize stages run in stage 1 and only
#                         the post-quantize stages run in stage 3.
#   --workdir   Directory for intermediate .pt2 / .pte / checkpoint files
#               (default: <script dir>/artifacts).
#
# Environment variables:
#   PYTHON      Python interpreter to use (default: python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
EXAMPLE="minimal"
WORKDIR="${SCRIPT_DIR}/artifacts"
PYTHON="${PYTHON:-python}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)
            EXAMPLE="$2"
            shift 2
            ;;
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        -h|--help)
            head -n 30 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$EXAMPLE" != "minimal" && "$EXAMPLE" != "sliced" ]]; then
    echo "Error: --example must be 'minimal' or 'sliced' (got '$EXAMPLE')" >&2
    exit 1
fi

mkdir -p "$WORKDIR"

echo "================================================================"
echo "  QAT pipeline split example -- mode: $EXAMPLE"
echo "  Intermediate artifacts: $WORKDIR"
echo "================================================================"
echo ""

# ------------------------------------------------------------------ Stage 1/4
echo "--- Stage 1/4: capture up to the quantize boundary ---"
echo ""
echo "  \$ $PYTHON 1_prepare.py --example $EXAMPLE --workdir $WORKDIR"
echo ""
"$PYTHON" "${SCRIPT_DIR}/1_prepare.py" --example "$EXAMPLE" --workdir "$WORKDIR"
echo ""
echo "  Stage 1 complete."
echo ""

# ------------------------------------------------------------------ Stage 2/4
echo "--- Stage 2/4: arbitrary QAT (+ checkpoint save/restore) ---"
echo ""
echo "  \$ $PYTHON 2_qat.py --example $EXAMPLE --workdir $WORKDIR"
echo ""
"$PYTHON" "${SCRIPT_DIR}/2_qat.py" --example "$EXAMPLE" --workdir "$WORKDIR"
echo ""
echo "  Stage 2 complete."
echo ""

# ------------------------------------------------------------------ Stage 3/4
echo "--- Stage 3/4: lower quantized graph to .pte (with the export recipe) ---"
echo ""
echo "  \$ $PYTHON 3_lower.py --example $EXAMPLE --workdir $WORKDIR"
echo ""
"$PYTHON" "${SCRIPT_DIR}/3_lower.py" --example "$EXAMPLE" --workdir "$WORKDIR"
echo ""
echo "  Stage 3 complete."
echo ""

# ------------------------------------------------------------------ Stage 4/4
echo "--- Stage 4/4: run model.pte through the ExecuTorch runtime ---"
echo ""
echo "  \$ $PYTHON 4_run.py --workdir $WORKDIR"
echo ""
"$PYTHON" "${SCRIPT_DIR}/4_run.py" --workdir "$WORKDIR"
echo ""
echo "  Stage 4 complete."
echo ""

# ----------------------------------------------------------------- Summary
PTE_FILE="${WORKDIR}/model.pte"
echo "================================================================"
echo "  Done.  Final artifact:"
echo ""
ls -lh "$PTE_FILE"
echo ""
echo "  Intermediate files:"
ls -lh "${WORKDIR}/"
echo "================================================================"
