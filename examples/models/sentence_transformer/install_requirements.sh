#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXECUTORCH_ROOT="${SCRIPT_DIR}/../../.."

echo "=================================================="
echo "Installing dependencies for sentence_transformer"
echo "=================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Using Python version: $PYTHON_VERSION"

# Install core dependencies
echo ""
echo "Installing Python dependencies..."
pip install transformers tokenizers scikit-learn numpy

echo ""
echo "✅ Dependencies installed successfully!"
echo ""

# Check if ExecuTorch is installed
if python -c "import executorch" 2>/dev/null; then
    echo "✅ ExecuTorch is already installed"
else
    echo "⚠️  ExecuTorch is not installed"
    echo ""
    echo "To install ExecuTorch, run from the repo root:"
    echo "  ./install_requirements.sh"
    echo ""
    echo "Or install manually:"
    echo "  pip install executorch"
fi

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Export a model:"
echo "     cd ${SCRIPT_DIR}"
echo "     python export_sentence_transformer.py --backend xnnpack"
echo ""
echo "  2. Validate the export:"
echo "     python compare_embeddings.py --model-path sentence_transformer_export/model.pte"
echo ""
echo "  3. Benchmark performance:"
echo "     python benchmark_backends.py"
echo ""
