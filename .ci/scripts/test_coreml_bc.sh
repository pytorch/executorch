#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test backward compatibility of CoreML runtime with older pte files.
# This script:
# 1. Checks out an old version of ExecuTorch
# 2. Exports a toy model to pte format using the old version
# 3. Checks out the current version (main)
# 4. Runs the old pte file using the current runtime via pybindings

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Create a conda environment with Python 3.10 for compatibility with old ET versions
# ET 1.0.0 only supports Python >=3.10,<3.13
CONDA_ENV_NAME="coreml_bc_test_env"
conda create -y -n "${CONDA_ENV_NAME}" python=3.10

# Use conda run to execute commands in the new environment
CONDA_RUN="conda run --no-capture-output -n ${CONDA_ENV_NAME}"

# The old hash to use for generating the pte file
# This should be a known stable release or commit that produces valid CoreML pte files
# Using ET 1.0.0 release (v1.0.0)
OLD_ET_HASH="${OLD_ET_HASH:-8c84780911e7f4e9eb19395181b04761392a4b56}"

# Model to export - using a simple model that is supported by CoreML
MODEL_NAME="${MODEL_NAME:-add}"

WORKING_DIR=$(pwd)
PTE_FILE="${WORKING_DIR}/bc_test_model.pte"

echo "=== CoreML Backward Compatibility Test ==="
echo "Old ET hash: ${OLD_ET_HASH}"
echo "Model: ${MODEL_NAME}"
echo "Working directory: ${WORKING_DIR}"

# Save the current HEAD so we can return to it
CURRENT_HEAD=$(git rev-parse HEAD)
echo "Current HEAD: ${CURRENT_HEAD}"

cleanup() {
    echo "Cleaning up..."
    cd "${WORKING_DIR}"
    git checkout "${CURRENT_HEAD}" --quiet || true
    rm -f "${PTE_FILE}"
    # Remove the conda environment
    conda env remove -n "${CONDA_ENV_NAME}" -y || true
}
trap cleanup EXIT

# Step 1: Checkout old version of ExecuTorch
echo "=== Step 1: Checking out old ET version (${OLD_ET_HASH}) ==="
git fetch origin "${OLD_ET_HASH}" --depth=1 || git fetch origin --unshallow || true
git checkout "${OLD_ET_HASH}"

# Step 2: Install the old version
echo "=== Step 2: Installing old ET version ==="
git submodule sync --recursive
git submodule update --init --recursive

# Install executorch
${CONDA_RUN} pip install --upgrade pip
${CONDA_RUN} python install_executorch.py

# Step 3: Export model
echo "=== Step 3: Exporting model with old ET version ==="
# Export a simple model using CoreML backend
${CONDA_RUN} python -c "
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.compiler import CoreMLBackend
import coremltools as ct

# Simple add model
class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = AddModel()
model.eval()

# Example inputs
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# Export with CoreML delegation
compile_specs = CoreMLBackend.generate_compile_specs(
    compute_precision=ct.precision.FLOAT32,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
)

ep = torch.export.export(model, (x, y))
edge_program = to_edge_transform_and_lower(
    ep,
    partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
)
exec_program = edge_program.to_executorch()

# Save the pte file
with open('${PTE_FILE}', 'wb') as f:
    exec_program.write_to_file(f)

print('Successfully exported model to ${PTE_FILE}')
"

# Verify pte file was created
if [[ ! -f "${PTE_FILE}" ]]; then
    echo "ERROR: Failed to create pte file"
    exit 1
fi
echo "PTE file created: $(ls -la "${PTE_FILE}")"

# Step 4: Checkout current version (main)
echo "=== Step 4: Checking out current ET version ==="
git checkout "${CURRENT_HEAD}"
git submodule sync --recursive
git submodule update --init --recursive

# Step 5: Install current version
echo "=== Step 5: Installing current ET version ==="
${CONDA_RUN} pip install --upgrade pip
${CONDA_RUN} python install_executorch.py

# Step 6: Run the old pte file
echo "=== Step 6: Running old pte with current runtime ==="
${CONDA_RUN} python -c "
import torch
from executorch.runtime import Runtime

print('Loading pte file: ${PTE_FILE}')
runtime = Runtime.get()
program = runtime.load_program('${PTE_FILE}')
method = program.load_method('forward')

# Create test inputs
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# Execute the model
print('Executing model...')
outputs = method.execute([x, y])
result = outputs[0]

print(f'Input x: {x}')
print(f'Input y: {y}')
print(f'Output: {result}')

# Verify the output is correct (x + y)
expected = x + y
if torch.allclose(result, expected, atol=1e-5):
    print('SUCCESS: Output matches expected result!')
else:
    print(f'FAILURE: Output does not match expected result')
    print(f'Expected: {expected}')
    print(f'Got: {result}')
    exit(1)
"

echo "=== CoreML Backward Compatibility Test PASSED ==="
