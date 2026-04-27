#!/bin/bash
# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Setup script for the Nordic AXON backend.
# Validates the environment and prints diagnostic information.

set -e

ERRORS=0

echo "=== Nordic AXON Backend Setup ==="
echo ""

# Check SDK_EDGE_AI_PATH
if [ -z "$SDK_EDGE_AI_PATH" ]; then
    echo "WARNING: SDK_EDGE_AI_PATH is not set."
    echo "  Set it to your Nordic sdk-edge-ai directory:"
    echo "  export SDK_EDGE_AI_PATH=/path/to/sdk-edge-ai"
    echo ""
    echo "  Without the SDK, TOSA lowering works but AXON compilation"
    echo "  (producing command buffer headers) will be skipped."
    echo ""
    SDK_STATUS="NOT SET"
    ERRORS=$((ERRORS + 1))
else
    if [ -d "$SDK_EDGE_AI_PATH" ]; then
        SDK_STATUS="OK ($SDK_EDGE_AI_PATH)"
        # Check for compiler lib
        SYSTEM=$(uname -s)
        case "$SYSTEM" in
            Linux)  LIB_NAME="libnrf-axon-nn-compiler-lib-amd64.so" ;;
            Darwin) LIB_NAME="libnrf-axon-nn-compiler-lib-arm64.dylib" ;;
            *)      LIB_NAME="nrf-axon-nn-compiler-lib-amd64.dll" ;;
        esac
        COMPILER_LIB="$SDK_EDGE_AI_PATH/tools/axon/compiler/bin/$SYSTEM/$LIB_NAME"
        if [ -f "$COMPILER_LIB" ]; then
            echo "  Compiler lib: $COMPILER_LIB"
        else
            echo "  WARNING: Compiler lib not found at: $COMPILER_LIB"
            SDK_STATUS="INCOMPLETE (missing compiler lib)"
        fi
    else
        echo "  WARNING: SDK_EDGE_AI_PATH directory does not exist: $SDK_EDGE_AI_PATH"
        SDK_STATUS="INVALID"
    fi
fi

# Check Python packages
echo ""
echo "Checking Python dependencies..."
MISSING=""
for pkg in cffi numpy yaml tosa; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "  $pkg: OK"
    else
        echo "  $pkg: MISSING"
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "Install missing packages:"
    echo "  pip install -r backends/nordic/requirements.txt"
fi

# Check ExecuTorch
echo ""
if python3 -c "import executorch" 2>/dev/null; then
    echo "ExecuTorch: OK"
else
    echo "ExecuTorch: NOT FOUND"
    echo "  Install ExecuTorch first — see the root README."
fi

# Check AXON backend
if python3 -c "from executorch.backends.nordic.axon import AxonBackend" 2>/dev/null; then
    echo "AXON backend: OK"
else
    echo "AXON backend: IMPORT FAILED"
fi

# Summary
echo ""
echo "=== Summary ==="
echo "  SDK_EDGE_AI_PATH: $SDK_STATUS"
echo "  Python deps:     $([ -z "$MISSING" ] && echo 'OK' || echo "MISSING:$MISSING")"
echo ""

# Run quick test
echo "Running quick import test..."
python3 -c "
from executorch.backends.nordic.axon import AxonBackend, AxonCompileSpec, AxonPartitioner
from executorch.backends.nordic.operator_support import AXON_SUPPORTED_OPS
print(f'  AxonBackend: OK')
print(f'  Supported ops: {len(AXON_SUPPORTED_OPS)}')
print('Setup complete.')
" 2>/dev/null || { echo "  Import test failed — check your ExecuTorch installation."; ERRORS=$((ERRORS + 1)); }

exit $ERRORS
