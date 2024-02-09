#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

#
# llama_executor_runner
#

cmake_minimum_required(VERSION 3.19)

project(llama_runner_example)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

add_compile_options("-Wall" "-Werror")

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find prebuilt libraries. executorch package should contain
# portable_ops_lib, etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party
)

# portable_ops_lib
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
gen_selected_ops("" "" "ON")
generate_bindings_for_kernels(
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml ""
)
gen_operators_lib("portable_ops_lib" portable_kernels executorch)

set(llama_runner_libs "-framework Foundation")

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(
  EXECUTORCH_SRCS_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

set(llama_schema_headers ${CMAKE_BINARY_DIR}/../../../schema/include/)
include(${EXECUTORCH_SRCS_FILE})
target_include_directories(
  bundled_program
  INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/../../../sdk/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../sdk/bundled_program
  ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
  ${EXECUTORCH_ROOT}/third-party/flatcc/include
  ${llama_schema_headers}
)

function(define_common_targets)
    foreach(aten IN ITEMS OFF) # ON OFF
        if(aten)
            set(aten_suffix "_aten")
        else()
            set(aten_suffix "")
        endif()

        message(">>>>>> CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

        set(EXPORTED_HEADERS_RUNNER
            "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/runner/util.h"
        )

        # Define a CMake target using runtime.cxx_library
        add_library(runner${aten_suffix}
            "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
        )

        # Specify exported headers
        target_include_directories(runner${aten_suffix} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/runner
            ${CMAKE_CURRENT_SOURCE_DIR}/sampler
            ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
        )

        get_target_property(runner_include_dirs runner${aten_suffix} INCLUDE_DIRECTORIES)
        message(">>>>Include Directories for runner_lib: ${runner_include_dirs}")


        target_sources(runner${aten_suffix} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
            ${EXPORTED_HEADERS_RUNNER}
        )

        add_library(sampler${aten_suffix}
            "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
        )

        target_include_directories(sampler${aten_suffix} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/sampler
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
        )


        target_sources(sampler${aten_suffix} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
        )


        add_library(tokenizer${aten_suffix}
            "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
        )

        target_include_directories(tokenizer${aten_suffix} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
        )


        target_sources(tokenizer${aten_suffix} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
        )


        # Add preprocessor flags if aten is true
        if(aten)
            target_compile_definitions(runner${aten_suffix} PRIVATE
                -DUSE_ATEN_LIB
            )
        endif()

        # Specify visibility
        target_compile_options(runner${aten_suffix} PRIVATE
            -fvisibility=hidden
        )

        # Specify exported dependencies
        target_link_libraries(runner${aten_suffix} PRIVATE
            //executorch/backends/xnnpack:xnnpack_backend
            //executorch/examples/models/llama2/sampler:sampler${aten_suffix}
            //executorch/examples/models/llama2/tokenizer:tokenizer${aten_suffix}
            //executorch/examples/models/llama2/runner:runner${aten_suffix}
            //executorch/extension/evalue_util:print_evalue${aten_suffix}
            //executorch/extension/runner_util:managed_tensor${aten_suffix}
            //executorch/extension/module:module${aten_suffix}
            //executorch/kernels/quantized:generated_lib${aten_suffix}
            //executorch/kernels/portable:${aten_suffix}generated_lib_all_ops
            //executorch/runtime/core/exec_aten:lib${aten_suffix}
            //executorch/runtime/kernel:kernel_includes
        )

        # Specify external dependencies if aten is true
        #if(aten)
            target_link_libraries(runner${aten_suffix} PRIVATE
                libtorch
            )
        #endif()

        # Set target properties, visibility, etc. as needed

    endforeach()
endfunction()

define_common_targets()

set(_llama_executor_runner__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
)
message(">>>>>_llama_executor_runner__srcs: ${_llama_executor_runner__srcs}")

#list(TRANSFORM _llama_executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_executable(llama_executor_runner ${_llama_executor_runner__srcs})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(FLATCC_LIB flatcc_d)
else()
    set(FLATCC_LIB flatcc)
endif()

target_link_libraries(llama_executor_runner bundled_program
                                          executorch
                                          gflags
                                          etdump
                                          ${FLATCC_LIB}
                                          portable_ops_lib
                                          ${llama_runner_libs})
target_compile_options(llama_executor_runner PUBLIC ${_common_compile_options})
