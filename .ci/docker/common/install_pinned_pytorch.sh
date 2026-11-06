#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Called directly or through .ci/scripts/install_pinned_pytorch.sh.
# Activate a venv/conda environment first, or set PYTHON_EXECUTABLE.
set -euo pipefail

dedupe_macos_loader_path_rpaths() {
  if [[ "$(uname)" != "Darwin" ]]; then
    return
  fi

  local torch_lib_dir
  torch_lib_dir=$(cd / && "${python_executable}" -c "import importlib.util; print(importlib.util.find_spec('torch').submodule_search_locations[0])")/lib

  if [[ -z "${torch_lib_dir}" || ! -d "${torch_lib_dir}" ]]; then
    return
  fi

  local torch_libs=(
    "libtorch_cpu.dylib"
    "libtorch.dylib"
    "libc10.dylib"
  )

  for lib_name in "${torch_libs[@]}"; do
    local lib_path="${torch_lib_dir}/${lib_name}"
    if [[ ! -f "${lib_path}" ]]; then
      continue
    fi

    local removed=0
    # Repeatedly remove the @loader_path rpath entries until none remain.
    while install_name_tool -delete_rpath @loader_path "${lib_path}" 2>/dev/null; do
      removed=1
    done

    if [[ "${removed}" == "1" ]]; then
      install_name_tool -add_rpath @loader_path "${lib_path}" || true
    fi
  done
}

verify_only=false
if [[ "${1:-}" == "--verify-only" ]]; then
  verify_only=true
  shift
fi
if [[ $# -gt 1 || "${1:-}" == -* ]]; then
  echo "Usage: $0 [--verify-only] [EXECUTORCH_DIRECTORY]" >&2
  exit 2
fi
executorch_dir=$(cd "${1:-.}" && pwd)
python_executable=${PYTHON_EXECUTABLE:-python}
# Resolve the interpreter before entering the temporary source directory.
python_executable=$("${python_executable}" -c 'import sys; print(sys.executable)')
system_name=$(uname)
case "${system_name}" in
  Linux | Darwin) ;;
  *) echo "Unsupported system: ${system_name}" >&2; exit 1 ;;
esac

torch_ref=${TORCH_VERSION:-$(cat "${executorch_dir}/.ci/docker/ci_commit_pins/pytorch.txt")}
torch_ref=$(printf '%s' "${torch_ref}" | tr -d '[:space:]')
git check-ref-format --branch "${torch_ref}" >/dev/null
installed_pin=$(cd / && "${python_executable}" -c \
  'import torch; print(torch.version.git_version or "")' 2>/dev/null || true)
expected_pin=${torch_ref}

verify_pin() {
  if [[ "${installed_pin}" != "${expected_pin}" ]]; then
    echo "PyTorch pin mismatch: expected ${expected_pin}, found ${installed_pin:-none}" >&2
    return 1
  fi
  (cd / && "${python_executable}" -c '
import torch
assert torch._C.has_lapack, "PyTorch was built without LAPACK"
torch.linalg.qr(torch.randn(4, 4))
')
}

install_domains() {
  "${python_executable}" -m pip install numpy pillow
  "${python_executable}" -m pip install --no-build-isolation --no-deps \
    'git+https://github.com/pytorch/audio.git@release/2.11' \
    'git+https://github.com/pytorch/vision.git@release/0.29'
}

if [[ ! "${torch_ref}" =~ ^[0-9a-f]{40}$ ]]; then
  expected_pin=$(git ls-remote --exit-code https://github.com/pytorch/pytorch.git \
    "${torch_ref}" "${torch_ref}^{}" | awk '
      { sub(/\^\{\}$/, "", $2); pins[$2] = $1 }
      END { for (ref in pins) print pins[ref] }
    ' | sort -u) || expected_pin=""
  if [[ ! "${expected_pin}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Cannot resolve PyTorch ref to one commit: ${torch_ref}" >&2
    exit 1
  fi
fi

if [[ "${installed_pin}" == "${expected_pin}" || "${verify_only}" == true ]]; then
  verify_pin
  echo "PyTorch pin ${expected_pin} is already installed"
  if [[ "${verify_only}" == false ]]; then
    dedupe_macos_loader_path_rpaths
    install_domains
  fi
  exit 0
fi

build_root=$(mktemp -d "${TMPDIR:-/tmp}/executorch-pytorch.XXXXXXXX")
trap '
  status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Failed to prepare the replacement PyTorch wheel." >&2
    echo "Existing torch and domain packages have not been removed." >&2
  fi
  rm -rf "${build_root}"
  exit "${status}"
' EXIT
git clone --filter=blob:none --no-checkout https://github.com/pytorch/pytorch.git "${build_root}/pytorch"
cd "${build_root}/pytorch"
git fetch --depth=1 origin "${expected_pin}"
git checkout --detach FETCH_HEAD

cached=false
if [[ "${system_name}" == Darwin ]]; then
  platform=$("${python_executable}" -c 'import sysconfig; import platform; v=platform.mac_ver()[0].split(".")[0]; platform=sysconfig.get_platform().split("-"); platform[1]=f"{v}_0"; print("_".join(platform))')
  python_version=$("${python_executable}" -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  torch_release=$(cat version.txt)
  torch_short_hash=$(git rev-parse --short=7 HEAD)
  torch_wheel_path="cached_artifacts/pytorch/executorch/pytorch_wheels/${system_name}/${python_version}"
  torch_wheel_name="torch-${torch_release}+git${torch_short_hash}-cp${python_version}-cp${python_version}-${platform}.whl"
  cached_torch_wheel="https://gha-artifacts.s3.us-east-1.amazonaws.com/${torch_wheel_path}/${torch_wheel_name//+/%2B}"
  if "${python_executable}" -m pip download --no-deps --dest dist "${cached_torch_wheel}"; then
    cached=true
  fi
fi

if [[ "${cached}" == false ]]; then
  echo "Building PyTorch at ${expected_pin}"
  git submodule update --init --recursive
  if [[ "$(uname -m)" == aarch64 ]]; then
    export BUILD_IGNORE_SVE_UNAVAILABLE=1
  fi
  if [[ "${system_name}" == Linux ]]; then
    export _GLIBCXX_USE_CXX11_ABI=1
  fi
  if [[ -n "${PYTORCH_BUILD_MAX_JOBS:-}" ]]; then
    export MAX_JOBS="${PYTORCH_BUILD_MAX_JOBS}"
  fi
  build_venv="${build_root}/build-env"
  "${python_executable}" -m venv --system-site-packages "${build_venv}"
  # Keep in sync with pytorch/pyproject.toml [build-system].requires.
  # Do not install pip cmake: it hides the image's MKL/libomp search paths.
  "${build_venv}/bin/python" -m pip install build "scikit-build-core>=1.0" \
    ninja "packaging>=24.2" "typing-extensions>=4.10.0" pyyaml six numpy
  CMAKE_CXX_SCAN_FOR_MODULES=OFF USE_DISTRIBUTED=1 \
    "${build_venv}/bin/python" -m build --wheel --no-isolation
fi

shopt -s nullglob
wheels=(dist/*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected one PyTorch wheel, found ${#wheels[@]}" >&2
  exit 1
fi
if [[ "${system_name}" == Darwin ]]; then
  if [[ "$(basename "${wheels[0]}")" != "${torch_wheel_name}" ]]; then
    echo "Built PyTorch wheel does not match cache key: ${torch_wheel_name}" >&2
    exit 1
  fi
fi

trap 'rm -rf "${build_root}"' EXIT
# Remove the old builds explicitly so equal package versions also replace
# different source commits, and domain dependencies cannot replace the pin.
"${python_executable}" -m pip uninstall -y torch torchaudio torchvision
"${python_executable}" -m pip install "${wheels[0]}"

dedupe_macos_loader_path_rpaths
install_domains
installed_pin=$(cd / && "${python_executable}" -c 'import torch; print(torch.version.git_version or "")')
verify_pin

# Preserve the existing macOS wheel cache on AWS CI runners.
if [[ "${system_name}" == Darwin && "${cached}" == false && "${GITHUB_ACTIONS:-}" == true && -z "${GITHUB_RUNNER:-}" ]] && command -v aws >/dev/null; then
  aws s3 cp "${wheels[0]}" "s3://gha-artifacts/${torch_wheel_path}/${torch_wheel_name}"
fi
sccache --show-stats || true
