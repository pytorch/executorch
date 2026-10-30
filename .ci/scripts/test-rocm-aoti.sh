#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.14}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
PYTORCH_ROCM_INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm${ROCM_VERSION}}"
ROCM_CI_TMP_ROOT="${RUNNER_TEMP:-/tmp}"
mkdir -p "${ROCM_CI_TMP_ROOT}" 2>/dev/null || ROCM_CI_TMP_ROOT=/tmp
ROCM_CI_TMPDIR="$(mktemp -d "${ROCM_CI_TMP_ROOT}/executorch-rocm-ci.XXXXXX")"
trap 'rm -rf "${ROCM_CI_TMPDIR}"' EXIT

export ROCM_PATH
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="${ROCM_CI_TMPDIR}/inductor-cache"
export TORCHINDUCTOR_COMPILE_THREADS=1

source .ci/scripts/install-rocm-dependencies.sh
EXECUTORCH_BUILD_MINIMAL=1 \
  python -m pip install --editable . --no-build-isolation

python - <<'PY'
import torch
import torchao
import triton

assert torch.version.hip is not None, "PyTorch is not a ROCm build"
assert torch.version.cuda is None, "PyTorch unexpectedly reports a CUDA runtime"
assert torch.cuda.is_available(), "No AMD GPU is visible through PyTorch"

device = torch.cuda.get_device_properties(0)
arch = device.gcnArchName.split(":", 1)[0]
assert arch == "gfx950", f"Expected gfx950, got {device.gcnArchName}"
assert device.warp_size == 64, f"Expected wave64, got {device.warp_size}"
assert arch in torch.cuda.get_arch_list(), (
    f"{arch} is not supported by this PyTorch build: {torch.cuda.get_arch_list()}"
)

print("PyTorch:", torch.__version__)
print("TorchAO:", torchao.__version__)
print("ROCm:", torch.version.hip)
print("Triton:", triton.__version__)
print("Device:", device.name, device.gcnArchName)
print("Architectures:", torch.cuda.get_arch_list())
PY


python -m examples.cuda.scripts.export_amd_pointwise --output-dir "${ROCM_CI_TMPDIR}/pointwise"
