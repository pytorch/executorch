#!/bin/bash
# Build the PyTorch::ExecuTorch CMSIS Pack
#
# This script packages the ExecuTorch sources into a CMSIS Pack.
# It expects sources to already be in executorch-pack/src/
#
# Usage:
#   ./build_pack.sh [OUTPUT_DIR] [VERSION]
#
# Arguments:
#   OUTPUT_DIR   Output directory for pack file (default: ./out/packs)
#   VERSION      Pack version (default: 0.6.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_ROOT="$(dirname "$SCRIPT_DIR")"

# Arguments
OUTPUT_DIR="${1:-$PACK_ROOT/out/packs}"
PACK_VERSION="${2:-1.1.0-rc1}"

# Get and increment build number
BUILD_NUMBER=$("$SCRIPT_DIR/increment_build_number.sh")
PACK_VERSION_WITH_BUILD="${PACK_VERSION}-build.${BUILD_NUMBER}"

# Source directory (must exist with sources)
SRC_DIR="$PACK_ROOT/src"

echo "=============================================="
echo "Building PyTorch::ExecuTorch CMSIS Pack"
echo "=============================================="
echo "Source directory: $SRC_DIR"
echo "Pack version: $PACK_VERSION_WITH_BUILD"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Validate source directory
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: Source directory not found: $SRC_DIR"
    echo "Run docker_pack_workflow.sh first to populate sources."
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"

PACK_BUILD="$OUTPUT_DIR/PyTorch.ExecuTorch.$PACK_VERSION_WITH_BUILD"
rm -rf "$PACK_BUILD"
mkdir -p "$PACK_BUILD"

# Step 1: Copy sources to pack structure
echo "=== Step 1: Copying sources ==="
# Use rsync to follow symlinks but avoid cycles
# The -L flag dereferences symlinks, --safe-links skips dangerous symlinks
if command -v rsync &> /dev/null; then
    rsync -a --copy-links --safe-links "$SRC_DIR/" "$PACK_BUILD/" 2>/dev/null || {
        echo "  rsync had some warnings (likely cyclic symlinks), continuing..."
    }
else
    # Fallback to cp, ignoring errors from cyclic links
    cp -rL "$SRC_DIR"/* "$PACK_BUILD/" 2>/dev/null || {
        echo "  cp had some warnings (likely cyclic symlinks), continuing..."
    }
fi

# Remove any broken or cyclic symlinks that may have been created
find "$PACK_BUILD" -type l -delete 2>/dev/null || true

# Step 2: Generate RegisterAllKernels.cpp with all operator registrations
echo "=== Step 2: Generating RegisterAllKernels.cpp ==="
if [[ -f "$SCRIPT_DIR/generate_register_all_kernels.py" ]]; then
    python3 "$SCRIPT_DIR/generate_register_all_kernels.py" \
        --source-dir "$PACK_BUILD" \
        --output "$PACK_BUILD/src/registration/RegisterAllKernels.cpp" || {
        echo "ERROR: Failed to generate RegisterAllKernels.cpp"
        exit 1
    }
    echo "Generated RegisterAllKernels.cpp with all operator registrations"
else
    echo "WARNING: generate_register_all_kernels.py not found, using existing RegisterAllKernels.cpp"
fi

# Step 3: Generate PDSC from template using the generator script
echo "=== Step 3: Generating PDSC with operator components ==="
TEMPLATE="$PACK_ROOT/templates/PyTorch.ExecuTorch.pdsc.tpl"
PDSC_OUT="$PACK_BUILD/PyTorch.ExecuTorch.pdsc"
COMPONENTS_FILE="$PACK_ROOT/build/components.xml"
mkdir -p "$PACK_ROOT/build"

# Use the generator to populate the template with all operator components
if [[ -f "$SCRIPT_DIR/generate_components.py" ]] && [[ -f "$TEMPLATE" ]]; then
    python3 "$SCRIPT_DIR/generate_components.py" \
        --source-dir "$PACK_BUILD" \
        --template "$TEMPLATE" \
        --pdsc-output "$PDSC_OUT" \
        --output "$COMPONENTS_FILE" \
        --version "$PACK_VERSION_WITH_BUILD" \
        --date "$(date +%Y-%m-%d)" || {
        echo "ERROR: Failed to generate PDSC with components"
        exit 1
    }
    echo "Generated PDSC with operator components: $PDSC_OUT"
else
    echo "ERROR: Missing generator script or template"
    exit 1
fi

# Step 4: Copy static files (LICENSE, etc.)
echo "=== Step 4: Copying static files ==="
if [[ -d "$PACK_ROOT/contributions/add" ]]; then
    cp -r "$PACK_ROOT/contributions/add/"* "$PACK_BUILD/" 2>/dev/null || true
fi

# Copy LICENSE
if [[ -f "$PACK_ROOT/LICENSE" ]]; then
    cp "$PACK_ROOT/LICENSE" "$PACK_BUILD/"
fi

# Step 5: Create pack file
echo "=== Step 5: Creating pack file ==="
# Use absolute path for pack file
PACK_FILE="$(cd "$OUTPUT_DIR" && pwd)/PyTorch.ExecuTorch.$PACK_VERSION_WITH_BUILD.pack"
cd "$PACK_BUILD"

# Remove any previous pack file
rm -f "$PACK_FILE"

# Create zip (pack is just a zip file)
# Try zip first, fall back to python if not available
if command -v zip &> /dev/null; then
    zip -r "$PACK_FILE" . -x "*.DS_Store" -x ".git*" -x "*/generated/*" -x "*.py"
else
    echo "zip not found, using Python to create archive..."
    python3 -c "
import zipfile
import os
import sys

pack_file = '$PACK_FILE'
pack_dir = '.'

with zipfile.ZipFile(pack_file, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(pack_dir):
        # Skip hidden directories and generated directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'generated']
        for file in files:
            if file.startswith('.') or file == '.DS_Store' or file.endswith('.py'):
                continue
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, pack_dir)
            zf.write(filepath, arcname)
print(f'Created: {pack_file}')
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
