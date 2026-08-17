#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# This script is run before building ExecuTorch binaries

# Initialize submodules here instead of during checkout so we can use OpenSSL
# on Windows (schannel fails with SEC_E_ILLEGAL_MESSAGE on some gitlab hosts).
UNAME_S=$(uname -s)
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
  git -c http.sslBackend=openssl submodule update --init
else
  git submodule update --init
fi

# Clone nested submodules for tokenizers - this is a workaround for recursive
# submodule clone failing due to path length limitations on Windows. Eventually,
# we should update the core job in test-infra to enable long paths before
# checkout to avoid needing to do this.
pushd extension/llm/tokenizers
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
  git -c http.sslBackend=openssl submodule update --init
else
  git submodule update --init
fi
popd

if [[ "$(uname -m)" == "aarch64" ]]; then
  # On some Linux aarch64 systems, the "atomic" library is not found during linking.
  # To work around this, replace "atomic" with the literal ${ATOMIC_LIB} so the
  # build system uses the full path to the atomic library.
  file="extension/llm/tokenizers/third-party/sentencepiece/src/CMakeLists.txt"
  sed 's/list(APPEND SPM_LIBS "atomic")/list(APPEND SPM_LIBS ${ATOMIC_LIB})/' \
    "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"

  grep -n 'list(APPEND SPM_LIBS ${ATOMIC_LIB})' "$file" && \
    echo "the file $file has been modified for atomic to use full path"
fi

# A CPU row must say so, rather than relying on the builder having no CUDA toolkit installed. The build
# turns CUDA on when it detects one, so a builder that gains a toolkit would silently start producing a
# CPU wheel carrying the CUDA delegate. That already happened on Windows, where the image ships a toolkit
# on PATH and the resulting wheel failed to load its own extension.
#
# Stated as the inverse rule: anything that does not name a CUDA train this project supports is a CPU row.
# An allowlist of spellings was tried first and left a gap for every spelling nobody thought of, which is
# the same defect twice: testing only for empty let a row spelled "cpu" through, and listing "cpu" still
# leaves "cpu-aarch64", "rocm" and anything else the matrix generator emits.
#
# The supported trains come from install_utils.py, so both classifiers see the same list.
# A row that names CUDA and is not a supported train fails here rather than being
# rebadged as CPU. The wheel matrix comes from a different repository than this list, so
# when they drift the alternative is publishing a CPU wheel under a CUDA-named index with
# no error anywhere: setup.py cannot catch it, because the CPU option this script writes
# is read first and returns early.
CUDA_ROW=0
if [[ -n "${CU_VERSION:-${DESIRED_CUDA:-}}" ]]; then
    ROW_VALUE="${CU_VERSION:-${DESIRED_CUDA:-}}"
    # Windows builders run MSYS bash in a conda environment that has python.exe
    # and no python3, so either name alone breaks a platform. Mirrors
    # run_python_script.sh. Resolving up front instead of testing the exit status
    # keeps a missing interpreter a hard failure rather than a silent CPU build.
    PYTHON_BIN=$(command -v python3 || command -v python)
    row_classification=$("${PYTHON_BIN}" - "${ROW_VALUE}" <<'PY'
import re, sys
sys.path.insert(0, '.')
from install_utils import SUPPORTED_CUDA_VERSIONS
raw = sys.argv[1].strip().lower()
trains = {f'{major}{minor}' for major, minor in SUPPORTED_CUDA_VERSIONS}
# Decide by what the row NAMES, not by the digits it happens to contain. Reducing the whole
# value to digits classified rocm13.2 as CUDA and rebadged a row named plainly "cuda" as CPU.
match = re.fullmatch(r'cu(?:da)?[-_]?(.*)', raw)
if match is None:
    print('cpu')
else:
    digits = re.sub(r'[^0-9]', '', match.group(1))
    print('cuda' if digits in trains else 'unsupported')
PY
)
    if [[ "${row_classification}" == "cuda" ]]; then
        CUDA_ROW=1
    elif [[ "${row_classification}" == "unsupported" ]]; then
        echo "row '${ROW_VALUE}' names a CUDA train this project does not support." >&2
        echo "Add it to SUPPORTED_CUDA_VERSIONS in install_utils.py, and to" >&2
        echo "_CUDA_RUNTIME_PACKAGES and _CUDA_LIBRARY_DIRECTORIES in setup.py, or" >&2
        echo "stop building this row. Building it as CPU would publish a CPU wheel" >&2
        echo "under a CUDA-named index." >&2
        exit 1
    fi
fi

if [[ ${CUDA_ROW} -eq 0 ]]; then
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DEXECUTORCH_BUILD_CUDA=OFF"
    echo "CMAKE_ARGS=${CMAKE_ARGS}" >> "${GITHUB_ENV}"
    echo "row '${CU_VERSION:-${DESIRED_CUDA:-}}' names no supported CUDA train, building CPU-only"
else
    # A CUDA row must produce the CUDA libraries. Left at the default, the build only
    # turns CUDA on if it happens to detect a toolkit, so a builder without one produced
    # a wheel tagged for CUDA, carrying no CUDA library, while still declaring the CUDA
    # runtime packages. That installs cleanly and then reports the backend as
    # unregistered when a model runs. Asking for it explicitly stops the decision from
    # depending on detection.
    #
    # It does not turn a missing toolkit into a configure failure. The CUDA directory
    # requires the toolkit but gates its sources on a working compiler, so a builder that
    # has toolkit files and no usable nvcc still configures and simply compiles none of
    # them. That leniency is deliberate, for packaging jobs that cannot complete compiler
    # identification, so the row itself has to verify the libraries it expected are present.
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DEXECUTORCH_BUILD_CUDA=ON"
    echo "CMAKE_ARGS=${CMAKE_ARGS}" >> "${GITHUB_ENV}"
    echo "row '${CU_VERSION:-${DESIRED_CUDA:-}}' is a CUDA row, requiring the CUDA build"
fi

# On Windows, enable symlinks and re-checkout the current revision to create
# the symlinked src/ directory. This is needed to build the wheel.
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
    echo "Enabling symlinks on Windows"
    git config core.symlinks true
    git checkout -f HEAD

    # Windows wheels are CPU-only (build-wheels-windows.yml sets
    # with-cuda: disabled), but the Windows CI image ships a CUDA toolkit on
    # PATH, which makes setup.py auto-enable EXECUTORCH_BUILD_CUDA. That bakes a
    # CUDA _C into the CPU wheel, which then fails its DLL load in the
    # smoke test ("DLL load failed while importing _C"). Force a
    # CPU-only build.
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DEXECUTORCH_BUILD_CUDA=OFF"
    echo "CMAKE_ARGS=${CMAKE_ARGS}" >> "${GITHUB_ENV}"
fi

# Manually install build requirements because `python setup.py bdist_wheel` does
# not install them. TODO(dbort): Switch to using `python -m build --wheel`,
# which does install them. Though we'd need to disable build isolation to be
# able to see the installed torch package.

"${GITHUB_WORKSPACE}/${REPOSITORY}/install_requirements.sh" --example

# Enable VGF in pybind wheel builds when the platform-specific build input is
# available from pip.
if [[ "$UNAME_S" == "Linux" || "$UNAME_S" == "Darwin" ]]; then
  if python3 -m pip install -r \
  # The wheel rewrites recorded runtime paths after linking, which needs patchelf. It is declared
  # under build-system requires, but that is honoured only by a frontend with build isolation and
  # this path deliberately installs requirements itself, so without this the rewrite silently
  # does nothing and a CUDA wheel loses its route to the NVIDIA libraries beside it.
  python3 -m pip install patchelf || true
    "${GITHUB_WORKSPACE}/${REPOSITORY}/backends/arm/requirements-arm-vgf-runtime.txt"; then
    export EXECUTORCH_PYBIND_ENABLE_VGF=ON
    echo "EXECUTORCH_PYBIND_ENABLE_VGF=ON" >> "${GITHUB_ENV}"
  else
    echo "VGF build dependency unavailable on this platform; building without VGF"
  fi
fi

# Download Qualcomm QNN SDK on Linux x86_64 so the wheel build can include the
# QNN backend.  The SDK is large, so we download it here (outside CMake) rather
# than during cmake configure.
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  echo "Downloading Qualcomm QNN SDK..."
  QNN_SDK_ROOT=$(python3 \
    "${GITHUB_WORKSPACE}/${REPOSITORY}/backends/qualcomm/scripts/download_qnn_sdk.py" \
    --print-sdk-path)
  export QNN_SDK_ROOT
  echo "QNN_SDK_ROOT=${QNN_SDK_ROOT}" >> "${GITHUB_ENV}"
  echo "QNN SDK downloaded to ${QNN_SDK_ROOT}"
fi

# Provision the Vulkan SDK (glslc) and submodules ONLY when explicitly requested
# via EXECUTORCH_BUILD_VULKAN. The default wheel build leaves this unset, so it
# does no extra work (no submodule fetch, no SDK download) and is unaffected.
if [[ "${EXECUTORCH_BUILD_VULKAN:-0}" != "0" \
      && "${EXECUTORCH_BUILD_VULKAN:-OFF}" != "OFF" ]]; then
  echo "Initializing Vulkan backend third-party submodules..."
  VULKAN_SUBMODULES=(
    backends/vulkan/third-party/Vulkan-Headers
    backends/vulkan/third-party/volk
    backends/vulkan/third-party/VulkanMemoryAllocator
  )
  if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
    git -c http.sslBackend=openssl submodule update --init "${VULKAN_SUBMODULES[@]}"
    echo "Installing Vulkan SDK for Windows wheel build..."
    powershell -ExecutionPolicy Bypass -File .ci/scripts/setup-vulkan-windows-deps.ps1
  else
    git submodule update --init "${VULKAN_SUBMODULES[@]}"
    # Install glslc from conda-forge rather than the LunarG SDK: the manylinux
    # wheel image uses an old glibc where the SDK's prebuilt glslc cannot run
    # ("GLIBC_2.29 not found"). conda-forge's shaderc is built against an old
    # sysroot and runs there. Vulkan headers come from the submodules above and
    # volk dlopen()s the loader at runtime, so only glslc is needed to build.
    echo "Installing glslc (conda-forge shaderc) for Linux wheel build..."
    _glslc_prefix="${HOME}/.shaderc"
    conda create -y -p "${_glslc_prefix}" -c conda-forge shaderc
    export PATH="${_glslc_prefix}/bin:${PATH}"
    echo "${_glslc_prefix}/bin" >> "${GITHUB_PATH}"
    echo "glslc installed: $(command -v glslc)"
  fi
else
  # This is the Vulkan equivalent of the Windows CUDA force-off above (#20527).
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DEXECUTORCH_BUILD_VULKAN=OFF"
  echo "CMAKE_ARGS=${CMAKE_ARGS}" >> "${GITHUB_ENV}"
fi
