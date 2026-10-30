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


ROCM_POINTWISE_STATUS=0
python -m examples.cuda.scripts.export_amd_pointwise --output-dir "${ROCM_CI_TMPDIR}/pointwise" || ROCM_POINTWISE_STATUS=$?
python - "${ROCM_CI_TMPDIR}/pointwise" <<'PYCODE'
from pathlib import Path
import re
import sys
import torch
root = Path(sys.argv[1])
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
triple = f"amdgcn-amd-amdhsa--{arch}".encode()
bare = arch.encode()
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    print("POINTWISE_FILE", path.relative_to(root), path.stat().st_size, flush=True)
paths = [root / "amd_triton.pte", root / "aoti_cuda_blob.ptd", *root.rglob("*.hsaco")]
for path in paths:
    if path.is_file():
        data = path.read_bytes()
        tokens = sorted(set(re.findall(rb"gfx[0-9][0-9a-z:+\-]*", data)))[:10]
        print(
            "ARCHITECTURE_CHECK",
            path.name,
            len(data),
            "triple=",
            triple in data,
            "bare=",
            bare in data,
            "tokens=",
            tokens,
            flush=True,
        )
PYCODE
exit "${ROCM_POINTWISE_STATUS}"
