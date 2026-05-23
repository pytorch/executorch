# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standalone runner project, invoked from examples/riscv/run.sh as:
# ~~~
#   cmake -S examples/riscv/baremetal -B <build> \
#       -DEXECUTORCH_ROOT=<repo>                 \
#       -DRISCV_BAREMETAL_PTE=<path>.bpte        \
#       -DCMAKE_TOOLCHAIN_FILE=.../riscv{32,64}-unknown-elf-toolchain.cmake
# ~~~
# Mirrors examples/arm/executor_runner/standalone/CMakeLists.txt so the
# top-level executorch CMake has no reference to examples/riscv/.

cmake_minimum_required(VERSION 3.20)
project(riscv_executor_runner_baremetal LANGUAGES C CXX ASM)

get_filename_component(
  _default_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT
      "${_default_executorch_root}"
      CACHE PATH "Path to the ExecuTorch checkout"
  )
endif()
if(NOT EXISTS "${EXECUTORCH_ROOT}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project."
  )
endif()

set(RISCV_BAREMETAL_PTE
    ""
    CACHE FILEPATH "Path to the .bpte to embed in the baremetal runner"
)
if(NOT RISCV_BAREMETAL_PTE)
  message(
    FATAL_ERROR
      "RISCV_BAREMETAL_PTE not set; pass -DRISCV_BAREMETAL_PTE=<path> from run.sh"
  )
endif()

include("${EXECUTORCH_ROOT}/tools/cmake/common/preset.cmake")
if(NOT DEFINED EXECUTORCH_BUILD_PRESET_FILE)
  set(EXECUTORCH_BUILD_PRESET_FILE
      "${EXECUTORCH_ROOT}/tools/cmake/preset/riscv64_baremetal.cmake"
      CACHE PATH "Preset used when configuring the standalone baremetal runner"
  )
endif()
load_build_preset()
include("${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake")

add_subdirectory(
  "${EXECUTORCH_ROOT}" "${CMAKE_BINARY_DIR}/executorch" EXCLUDE_FROM_ALL
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_pte_header "${CMAKE_CURRENT_BINARY_DIR}/model_pte.h")
add_custom_command(
  OUTPUT "${_pte_header}"
  COMMAND
    "${Python3_EXECUTABLE}"
    "${EXECUTORCH_ROOT}/examples/arm/executor_runner/pte_to_header.py" --pte
    "${RISCV_BAREMETAL_PTE}" --outdir "${CMAKE_CURRENT_BINARY_DIR}" --outfile
    "model_pte.h" --section ".rodata.model_pte"
  DEPENDS "${RISCV_BAREMETAL_PTE}"
  COMMENT "Embedding ${RISCV_BAREMETAL_PTE} into model_pte.h"
  VERBATIM
)

# pte_to_header.py emits the byte array but not its length; the glue TU
# materialises the matching `model_pte_len` and is the only place the header is
# included (avoids a double-definition at link time).
file(
  WRITE "${CMAKE_CURRENT_BINARY_DIR}/model_pte_glue.cpp"
  "#include <stddef.h>\n#include \"model_pte.h\"\nextern \"C\" const size_t model_pte_len = sizeof(model_pte);\n"
)

add_executable(
  executor_runner_baremetal
  start.S executor_runner_baremetal.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/model_pte_glue.cpp" "${_pte_header}"
)
set_target_properties(
  executor_runner_baremetal PROPERTIES SUFFIX ".elf" LINKER_LANGUAGE CXX
)
target_include_directories(
  executor_runner_baremetal PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)
target_compile_options(
  executor_runner_baremetal PRIVATE -fno-exceptions -fno-rtti -fdata-sections
                                    -ffunction-sections
)
# --specs=picolibc.specs / -nostartfiles / -march / -mabi all come from the
# toolchain file; only the linker script (QEMU virt memory map) is target-
# specific here.
target_link_options(
  executor_runner_baremetal PRIVATE
  "-T${CMAKE_CURRENT_SOURCE_DIR}/riscv_virt.ld"
)

# gen_operators_lib / executorch_target_link_options_shared_lib attach INTERFACE
# --whole-archive options to portable_ops_lib (so the static-init
# kernel-registration TU survives DCE) and to executorch itself. Listing the
# libs once each is enough; an extra --whole-archive wrapper around them would
# include the same archive twice and double-register every op.
target_link_libraries(executor_runner_baremetal PRIVATE bundled_program)
if(TARGET portable_ops_lib)
  target_link_libraries(executor_runner_baremetal PRIVATE portable_ops_lib)
endif()
if(TARGET portable_kernels)
  target_link_libraries(executor_runner_baremetal PRIVATE portable_kernels)
endif()
