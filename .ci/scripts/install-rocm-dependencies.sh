#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if ! command -v conda >/dev/null; then
  echo "The ROCm CI image must provide conda for its C++ runtime libraries"
  exit 1
fi
conda install -y --override-channels -c conda-forge \
  gxx_linux-64=16.2.0 libstdcxx-ng=16.2.0 libgcc-ng=16.2.0 \
  sysroot_linux-64=2.28
ROCM_PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export CC="${ROCM_PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${ROCM_PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="${ROCM_PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

TORCH_VERSION="$(python -c 'from torch_pin import TORCH_VERSION; print(TORCH_VERSION)')"
python -m pip install "torch==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
  --index-url "${PYTORCH_ROCM_INDEX}"
ROCM_SDK_VERSION="$(python -c 'import rocm_sdk; print(rocm_sdk.__version__)')"
python -m pip install "rocm[devel]==${ROCM_SDK_VERSION}" \
  --index-url "${PYTORCH_ROCM_INDEX}"
ROCM_PATH="$(python -m rocm_sdk path --root)"
export ROCM_PATH
export ROCM_HOME="${ROCM_PATH}"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PYTHON_PREFIX}/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-dev.txt ninja
# ROCm 7.14 TorchAO wheels are Python-only. Build the pinned extension against
# the installed torch and HIP libraries instead of reusing a ROCm 7.2 binary.
USE_CPP=1 USE_CPU_KERNELS=1 MAX_JOBS="${MAX_JOBS:-8}" \
  python -m pip install --no-build-isolation --no-deps third-party/ao

python - <<'PY'
from pathlib import Path

import torch
import torchao
from packaging.version import Version

from install_requirements import TORCHAO_MIN_VERSION

assert torch.version.hip is not None, "PyTorch is not a ROCm build"
assert Version(torchao.__version__) >= Version(TORCHAO_MIN_VERSION)
libraries = list(Path(torchao.__file__).parent.glob("_C*.so"))
assert libraries, "TorchAO was built without its compiled extension"
for library in libraries:
    # TorchAO's import logs load failures and continues; CI must fail on them.
    torch.ops.load_library(str(library))
print("Loaded TorchAO extensions:", libraries)
PY
