#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the Linux wheel with the prebuilt C++ SDK, installs it into a clean
# environment, and builds a standalone C++ program against it via
# find_package(executorch). This is the signal that the shipped
# libexecutorch.so, headers, and CMake package config actually let a C++
# consumer link the ExecuTorch runtime with no source checkout, and that a
# separately built "coreless" backend shared library can register into the one
# runtime registry (the mechanism coalesced multi-backend execution relies on).

set -euxo pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_VENV="${REPO_ROOT}/.venv-sdk-build"
TEST_VENV="${REPO_ROOT}/.venv-sdk-test"
WORK_DIR="${REPO_ROOT}/cpp_sdk_consumer"

rm -rf "${BUILD_VENV}" "${TEST_VENV}" "${WORK_DIR}" "${REPO_ROOT}/dist" \
  "${REPO_ROOT}/pip-out"

# ---------------------------------------------------------------------------
# Build the wheel.
# ---------------------------------------------------------------------------
"${PYTHON_EXECUTABLE}" -m venv "${BUILD_VENV}"
# shellcheck source=/dev/null
source "${BUILD_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "cmake>=3.24,<4.0.0" \
  "numpy>=2.0.0" \
  packaging \
  pyyaml \
  setuptools \
  wheel \
  zstd \
  certifi \
  torch \
  torchvision \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
  --extra-index-url https://pypi.org/simple

(
  cd "${REPO_ROOT}"
  CMAKE_ARGS="${SDK_WHEEL_CMAKE_ARGS:-}" python setup.py bdist_wheel
)

WHEEL_FILE="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name 'executorch-*.whl' | head -1)"
test -n "${WHEEL_FILE}"

# ---------------------------------------------------------------------------
# Verify the SDK payload is present in the wheel.
# ---------------------------------------------------------------------------
python - "${WHEEL_FILE}" <<'PY'
import sys
import zipfile

wheel_file = sys.argv[1]
with zipfile.ZipFile(wheel_file) as wheel:
    names = set(wheel.namelist())

required = [
    "executorch/share/cmake/executorch-config.cmake",
    "executorch/utils/__init__.py",
    "executorch/include/executorch/runtime/executor/program.h",
    "executorch/include/executorch/extension/module/module.h",
]
missing = [name for name in required if name not in names]
if missing:
    raise AssertionError(f"{wheel_file} is missing SDK files: {missing}")

# The loader resolves the SONAME (libexecutorch.so.<major>), which is the only
# runtime library shipped. Without it the wheel has no linkable runtime.
versioned = [
    name
    for name in names
    if name.startswith("executorch/lib/libexecutorch.so.")
    and name.split("libexecutorch.so.")[1][:1].isdigit()
]
if not versioned:
    raise AssertionError(
        f"{wheel_file} has no versioned libexecutorch.so.<ver>; the SONAME "
        "library is missing and the runtime cannot load."
    )

# Headers whose implementation is not shipped must not be advertised.
forbidden = [
    "executorch/include/executorch/extension/module/bundled_module.h",
    "executorch/include/executorch/extension/flat_tensor/serialize/serialize.h",
]
present = [name for name in forbidden if name in names]
if present:
    raise AssertionError(f"{wheel_file} advertises unshipped APIs: {present}")

print("SDK payload OK")
PY

deactivate

# ---------------------------------------------------------------------------
# Install into a clean environment and link a C++ consumer against it.
# ---------------------------------------------------------------------------
"${PYTHON_EXECUTABLE}" -m venv "${TEST_VENV}"
# shellcheck source=/dev/null
source "${TEST_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "cmake>=3.24,<4.0.0"
# --no-deps: the C++ SDK link test does not need torch, proving the runtime is
# linkable standalone. (A plain install pulls declared deps; not needed here.)
python -m pip install --no-deps "${WHEEL_FILE}"

CMAKE_PREFIX_PATH="$(python -c 'import executorch.utils as u; print(u.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

cat > main.cpp <<'CPP'
#include <cstdio>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/named_data_map/merged_data_map.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::from_blob;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::get_backend_class;
using executorch::runtime::get_num_registered_backends;
using executorch::runtime::runtime_init;

int main() {
  runtime_init();
  printf("registered backends: %zu\n", get_num_registered_backends());
  // The delegate-only SDK ships no backends, so this must be null. What matters
  // is that the symbol links and resolves from libexecutorch.so.
  printf("stock backend lookup resolves: %d\n",
         get_backend_class("NonexistentBackend") == nullptr);

  // Exercise the advertised SDK surface so a broken header or missing symbol in
  // Module / Tensor / DataLoader is caught at compile+link time, not just the
  // runtime/backend headers.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorPtr tensor = from_blob(data.data(), {2, 2});
  printf("tensor numel: %zu\n", tensor->numel());

  // Exercise the .ptd data-map surface (FlatTensorDataMap) so those advertised
  // headers/symbols are compiled and linked from libexecutorch.so. Loading a
  // bogus buffer is expected to fail; we only need the symbol to resolve.
  {
    const uint8_t bogus_ptd[8] = {0};
    auto data_map_result = FlatTensorDataMap::load(
        std::make_unique<executorch::extension::BufferDataLoader>(
            bogus_ptd, sizeof(bogus_ptd))
            .get());
    printf(".ptd data-map load attempted (ok=%d)\n",
           data_map_result.ok());
  }

  // odr-use MergedDataMap so a dropped symbol fails the link, not just
  // a header include. An empty span is a valid no-op merge.
  {
    executorch::runtime::Span<const executorch::runtime::NamedDataMap*>
        maps;
    auto merged = executorch::extension::MergedDataMap::load(maps);
    printf("merged data-map load attempted (ok=%d)\n", merged.ok());
  }

  // Construct a Module from a tiny in-memory buffer via the buffer data loader.
  // Loading is expected to fail (not a real .pte); we only need this to
  // compile and link against the Module/DataLoader symbols in libexecutorch.so.
  const uint8_t fake_pte[8] = {0};
  Module module(std::make_unique<executorch::extension::BufferDataLoader>(
      fake_pte, sizeof(fake_pte)));
  (void)module.method_names();

  printf("PASS: linked executorch runtime + Module/Tensor/DataLoader from the "
         "wheel\n");
  return 0;
}
CPP

cat > CMakeLists.txt <<'CMAKE'
cmake_minimum_required(VERSION 3.19)
project(executorch_cpp_sdk_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(executorch CONFIG REQUIRED)
if(NOT EXECUTORCH_SDK_FOUND)
  message(FATAL_ERROR "EXECUTORCH_SDK_FOUND is false; the wheel C++ SDK is missing")
endif()
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE executorch::runtime)
CMAKE

cmake -S . -B build
cmake --build build
./build/consumer

# The runtime and the consumer must not pull in libtorch.
if ldd ./build/consumer | grep -Eiq "libtorch|libc10"; then
  echo "ERROR: consumer unexpectedly links libtorch/libc10" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Prove cross-shared-object registration: a "coreless" backend .so (no bundled
# runtime, register_backend undefined) registers into the runtime inside
# libexecutorch.so when loaded. This is the mechanism coalesced multi-backend
# execution depends on.
# ---------------------------------------------------------------------------
cat > mybackend.cpp <<'CPP'
#include <executorch/runtime/backend/interface.h>
using namespace executorch::runtime;
namespace {
struct MyBackend final : public BackendInterface {
  bool is_available() const override { return true; }
  Result<DelegateHandle*> init(BackendInitContext&, FreeableBuffer*,
                               ArrayRef<CompileSpec>) const override {
    return nullptr;
  }
  Error execute(BackendExecutionContext&, DelegateHandle*,
                Span<EValue*>) const override {
    return Error::Ok;
  }
  void destroy(DelegateHandle*) const override {}
};
MyBackend g_backend;
Backend g_id{"MyTestBackend", &g_backend};
static auto g_registered = register_backend(g_id);
} // namespace
CPP

cat >> CMakeLists.txt <<'CMAKE'
# Coreless: undefined ExecuTorch symbols resolve from libexecutorch.so at load.
add_library(mybackend SHARED mybackend.cpp)
target_link_options(mybackend PRIVATE "LINKER:--unresolved-symbols=ignore-all")
target_include_directories(mybackend PRIVATE
  $<TARGET_PROPERTY:executorch::runtime,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(mybackend PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)

add_executable(reg_consumer reg_main.cpp)
target_link_libraries(reg_consumer PRIVATE executorch::runtime ${CMAKE_DL_LIBS})
CMAKE

cat > reg_main.cpp <<'CPP'
#include <cstdio>
#include <dlfcn.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::get_backend_class;
using executorch::runtime::runtime_init;

int main(int argc, char** argv) {
  runtime_init();
  if (get_backend_class("MyTestBackend") != nullptr) {
    fprintf(stderr, "backend registered before load\n");
    return 1;
  }
  void* handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  if (get_backend_class("MyTestBackend") == nullptr) {
    fprintf(stderr, "backend NOT registered after load\n");
    return 1;
  }
  printf("PASS: coreless backend .so registered into libexecutorch.so\n");
  return 0;
}
CPP

cmake -S . -B build
cmake --build build
./build/reg_consumer "$(find build -name 'libmybackend.so' | head -1)"

# ---------------------------------------------------------------------------
# CUDA delegate check. Only runs when the wheel actually shipped the CUDA
# backend (i.e. a CUDA-enabled wheel); a CPU wheel skips this cleanly. Verifies
# the REAL shipped artifact: dlopen libexecutorch_cuda_backend.so and assert it
# registers "CudaBackend" into the runtime in libexecutorch.so.
SDK_LIB_DIR="$(python -c 'import os, executorch.utils as u; print(os.path.join(os.path.dirname(os.path.dirname(u.cmake_prefix_path)), "lib"))')"
CUDA_BACKEND_SO="${SDK_LIB_DIR}/libexecutorch_cuda_backend.so"
if [ ! -f "${CUDA_BACKEND_SO}" ] && [ "${EXPECT_CUDA:-0}" = "1" ]; then
  echo "EXPECT_CUDA=1 but libexecutorch_cuda_backend.so is missing" >&2
  exit 1
fi
if [ -f "${CUDA_BACKEND_SO}" ]; then
  echo "CUDA backend present; verifying it registers CudaBackend"
  cat > cuda_reg.cpp <<'CPP'
#include <cstdio>
#include <dlfcn.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::get_backend_class;
using executorch::runtime::runtime_init;

int main(int argc, char** argv) {
  runtime_init();
  if (get_backend_class("CudaBackend") != nullptr) {
    fprintf(stderr, "CudaBackend registered before load\n");
    return 1;
  }
  void* handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  if (get_backend_class("CudaBackend") == nullptr) {
    fprintf(stderr, "CudaBackend NOT registered after load\n");
    return 1;
  }
  printf("PASS: shipped libexecutorch_cuda_backend.so registered CudaBackend\n");
  return 0;
}
CPP
  cat >> CMakeLists.txt <<'CMAKE'
add_executable(cuda_reg cuda_reg.cpp)
target_link_libraries(cuda_reg PRIVATE executorch::runtime ${CMAKE_DL_LIBS})
CMAKE
  cmake -S . -B build
  cmake --build build
  # dlopen by absolute path; its $ORIGIN rpath resolves the co-shipped deps.
  ./build/cuda_reg "${CUDA_BACKEND_SO}"
else
  echo "CUDA backend not in wheel (CPU wheel); skipping CudaBackend check"
fi

echo "ALL C++ SDK WHEEL CHECKS PASSED"
