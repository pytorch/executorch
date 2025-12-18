#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Script to test sentence_transformer example export and validation
# This is used in CI to ensure the sentence_transformer example works correctly

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
EXAMPLE_DIR="examples/models/sentence_transformer"
TEST_OUTPUT_DIR="./sentence_transformer_ci_test"

echo "========================================="
echo "Testing sentence_transformer example"
echo "========================================="

# Navigate to example directory
cd "${EXAMPLE_DIR}"

# Clean up any previous test outputs
rm -rf "${TEST_OUTPUT_DIR}"
mkdir -p "${TEST_OUTPUT_DIR}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install transformers tokenizers scikit-learn numpy

# Test 1: Export with XNNPack backend
echo ""
echo "========================================="
echo "Test 1: Export with XNNPack backend"
echo "========================================="
python export_sentence_transformer.py \
    --model "${MODEL_NAME}" \
    --backend xnnpack \
    --output-dir "${TEST_OUTPUT_DIR}/xnnpack_export" \
    --max-seq-length 128

# Verify export succeeded
if [ ! -f "${TEST_OUTPUT_DIR}/xnnpack_export/model.pte" ]; then
    echo "ERROR: XNNPack export failed - model.pte not found"
    exit 1
fi
echo "✓ XNNPack export succeeded"

# Test 2: Export with CPU backend
echo ""
echo "========================================="
echo "Test 2: Export with CPU backend"
echo "========================================="
python export_sentence_transformer.py \
    --model "${MODEL_NAME}" \
    --backend cpu \
    --output-dir "${TEST_OUTPUT_DIR}/cpu_export" \
    --max-seq-length 128

# Verify export succeeded
if [ ! -f "${TEST_OUTPUT_DIR}/cpu_export/model.pte" ]; then
    echo "ERROR: CPU export failed - model.pte not found"
    exit 1
fi
echo "✓ CPU export succeeded"

# Test 3: Validate embeddings (XNNPack)
echo ""
echo "========================================="
echo "Test 3: Validate embeddings (XNNPack)"
echo "========================================="
python compare_embeddings.py \
    --model-path "${TEST_OUTPUT_DIR}/xnnpack_export/model.pte" \
    --model-name "${MODEL_NAME}" \
    --sentences "This is a test sentence for CI validation." \
    --max-length 128

echo "✓ XNNPack embedding validation passed"

# Test 4: Validate embeddings (CPU)
echo ""
echo "========================================="
echo "Test 4: Validate embeddings (CPU)"
echo "========================================="
python compare_embeddings.py \
    --model-path "${TEST_OUTPUT_DIR}/cpu_export/model.pte" \
    --model-name "${MODEL_NAME}" \
    --sentences "This is a test sentence for CI validation." \
    --max-length 128

echo "✓ CPU embedding validation passed"

# Test 5: Quick benchmark test (reduced iterations for CI)
echo ""
echo "========================================="
echo "Test 5: Quick benchmark test"
echo "========================================="
# Skip full benchmark in CI, just verify the tool runs
python benchmark_backends.py \
    --iterations 5 \
    --warmup-iterations 2 \
    --output-dir "${TEST_OUTPUT_DIR}/benchmark_test" \
    --max-seq-length 128

echo "✓ Benchmark tool runs successfully"

# Test 6: Create input bins tool
echo ""
echo "========================================="
echo "Test 6: Create input bins tool"
echo "========================================="
python create_input_bins.py \
    --text "CI test sentence for binary input generation." \
    --model "${MODEL_NAME}" \
    --max-length 128 \
    --output-dir "${TEST_OUTPUT_DIR}/input_bins_test"

# Verify binary files were created
if [ ! -f "${TEST_OUTPUT_DIR}/input_bins_test/input_ids.bin" ]; then
    echo "ERROR: input_ids.bin not created"
    exit 1
fi

if [ ! -f "${TEST_OUTPUT_DIR}/input_bins_test/attention_mask.bin" ]; then
    echo "ERROR: attention_mask.bin not created"
    exit 1
fi

echo "✓ Input bins creation succeeded"

# Clean up
echo ""
echo "Cleaning up test outputs..."
rm -rf "${TEST_OUTPUT_DIR}"

echo ""
echo "========================================="
echo "All tests passed! ✓"
echo "========================================="
echo ""
echo "Tests completed:"
echo "  ✓ XNNPack export"
echo "  ✓ CPU export"
echo "  ✓ XNNPack embedding validation"
echo "  ✓ CPU embedding validation"
echo "  ✓ Benchmark tool execution"
echo ""
