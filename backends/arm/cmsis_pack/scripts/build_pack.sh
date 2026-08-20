#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the PyTorch::ExecuTorch CMSIS Pack
#
# This script packages ExecuTorch sources into a CMSIS Pack (.pack file).
# It collects sources from the ExecuTorch repo tree and CMake build outputs,
# generates the PDSC manifest with per-operator components, and creates
# the final .pack archive.
#
# Usage:
#   ./build_pack.sh --executorch-root <path> --build-dir <path> \
#                   --version <ver> --output-dir <path>
#
# All arguments are required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_ROOT="$(cd "$SCRIPT_DIR/../../cmsis_pack" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
EXECUTORCH_ROOT=""
BUILD_DIR=""
PACK_VERSION=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --executorch-root) EXECUTORCH_ROOT="$2"; shift 2 ;;
        --build-dir)      BUILD_DIR="$2";        shift 2 ;;
        --version)        PACK_VERSION="$2";     shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXECUTORCH_ROOT" || -z "$BUILD_DIR" || -z "$PACK_VERSION" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --executorch-root <path> --build-dir <path> --version <ver> --output-dir <path>"
    exit 1
fi

echo "=============================================="
echo "Building PyTorch::ExecuTorch CMSIS Pack"
echo "=============================================="
echo "ExecuTorch root: $EXECUTORCH_ROOT"
echo "CMake build dir: $BUILD_DIR"
echo "Pack version:    $PACK_VERSION"
echo "Output dir:      $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Prepare staging area
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
PACK_BUILD="$OUTPUT_DIR/PyTorch.ExecuTorch.$PACK_VERSION"
rm -rf "$PACK_BUILD"
mkdir -p "$PACK_BUILD"

# Step 1: Copy sources from the repo / build tree into the pack layout
echo "=== Step 1: Copying sources ==="
"$SCRIPT_DIR/copy_sources.sh" \
    --executorch-root "$EXECUTORCH_ROOT" \
    --build-dir "$BUILD_DIR" \
    --pack-staging "$PACK_BUILD"

# Step 2: Generate RegisterAllKernels.cpp with #ifdef-guarded registrations
echo "=== Step 2: Generating RegisterAllKernels.cpp ==="
python3 "$SCRIPT_DIR/generate_register_all_kernels.py" \
    --source-dir "$PACK_BUILD" \
    --output "$PACK_BUILD/src/registration/RegisterAllKernels.cpp"

# Step 3: Generate PDSC from template
echo "=== Step 3: Generating PDSC with operator components ==="
TEMPLATE="$PACK_ROOT/templates/PyTorch.ExecuTorch.pdsc.tpl"
PDSC_OUT="$PACK_BUILD/PyTorch.ExecuTorch.pdsc"

python3 "$SCRIPT_DIR/generate_components.py" \
    --source-dir "$PACK_BUILD" \
    --template "$TEMPLATE" \
    --pdsc-output "$PDSC_OUT" \
    --output "$OUTPUT_DIR/components.xml" \
    --version "$PACK_VERSION" \
    --date "$(date +%Y-%m-%d)"

# Step 4: Copy static files (LICENSE, docs)
echo "=== Step 4: Copying static files ==="
if [[ -d "$PACK_ROOT/contributions/add" ]]; then
    cp -r "$PACK_ROOT/contributions/add/"* "$PACK_BUILD/" 2>/dev/null || true
fi
# Use the repo-root LICENSE (BSD-3-Clause from Meta)
if [[ -f "$EXECUTORCH_ROOT/LICENSE" ]]; then
    cp "$EXECUTORCH_ROOT/LICENSE" "$PACK_BUILD/"
fi

# Step 5: Create .pack archive (a zip file)
echo "=== Step 5: Creating pack file ==="
PACK_FILE="$(cd "$OUTPUT_DIR" && pwd)/PyTorch.ExecuTorch.$PACK_VERSION.pack"
rm -f "$PACK_FILE"
cd "$PACK_BUILD"

if command -v zip &> /dev/null; then
    zip -r "$PACK_FILE" . -x "*.DS_Store" -x ".git*" -x "*/generated/*" -x "*.py"
else
    python3 -c "
import zipfile, os
with zipfile.ZipFile('$PACK_FILE', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'generated']
        for f in files:
            if f.startswith('.') or f == '.DS_Store' or f.endswith('.py'):
                continue
            p = os.path.join(root, f)
            zf.write(p, os.path.relpath(p, '.'))
print(f'Created: $PACK_FILE')
"
fi

echo ""
echo "=============================================="
echo "Pack build complete!"
echo "=============================================="
echo "Pack file: $PACK_FILE"
echo "Pack size: $(du -h "$PACK_FILE" | cut -f1)"
echo ""
echo "To install: cpackget add $PACK_FILE"
