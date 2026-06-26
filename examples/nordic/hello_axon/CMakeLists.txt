# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Hello AXON — minimal ExecuTorch + AXON NPU inference.
#
# Build:
#   source nrf-connect-sdk-env.sh
#   west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/hello_axon \
#     --no-sysbuild -- \
#     -DZEPHYR_EXTRA_MODULES="<executorch_path>;<sdk_edge_ai_path>"

cmake_minimum_required(VERSION 3.24)

# Skip install rules — avoids ExecuTorch export dependency issues with Zephyr
set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

# Source files: main.c (pure C entry) + inference.cpp (C++ ExecuTorch runner)
target_sources(app PRIVATE
    src/main.c
    src/inference.cpp
)

# AXON delegate (from our ExecuTorch fork's backends/nordic/runtime/)
if(CONFIG_NRF_AXON)
    if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
        target_sources(app PRIVATE
            ${ZEPHYR_EXECUTORCH_MODULE_DIR}/backends/nordic/runtime/AxonBackend.cpp
            ${ZEPHYR_EXECUTORCH_MODULE_DIR}/backends/nordic/runtime/axon_op_extensions.c
        )
    endif()
endif()

# Include app source dirs + generated headers
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/generated
)

# Stub axon_subgraphs_table.h if not generated yet
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/axon_subgraphs_table.h)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/axon_subgraphs_table.h
"/* Stub — run export_model.py to generate real AXON subgraph headers. */\n"
"#pragma once\n"
"#include \"axon/nrf_axon_platform.h\"\n"
"#include \"drivers/axon/nrf_axon_nn_infer.h\"\n"
"#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1\n"
"typedef struct {\n"
"    const char *name;\n"
"    const nrf_axon_nn_compiled_model_s *model;\n"
"} axon_subgraph_entry_t;\n"
"#define AXON_SUBGRAPHS_COUNT 0\n"
"static const axon_subgraph_entry_t axon_subgraphs[1] = {{0}};\n")
endif()

# ExecuTorch setup
if(CONFIG_EXECUTORCH)
    if(NOT DEFINED EXECUTORCH_DIR)
        if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
            set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
        endif()
    endif()

    set(EXECUTORCH_ROOT ${EXECUTORCH_DIR})
    include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)

    # Kernel registry: 27 prim + 2 selective + 19 quantized = 48 → use 64
    target_compile_definitions(app PRIVATE MAX_KERNEL_NUM=64)
    if(TARGET executorch_core)
        target_compile_definitions(executorch_core PRIVATE MAX_KERNEL_NUM=64)
    endif()

    # Link ExecuTorch (provides include paths + core runtime)
    target_link_libraries(app PRIVATE libexecutorch)

    # Portable kernels (for un-delegated ops)
    if(TARGET portable_kernels)
        executorch_target_link_options_shared_lib(portable_kernels)
        target_link_libraries(app PRIVATE portable_kernels)
    endif()

    # Quantized ops (for q/dq at AXON delegation boundaries)
    if(TARGET quantized_ops_lib)
        executorch_target_link_options_shared_lib(quantized_ops_lib)
        target_link_libraries(app PRIVATE quantized_ops_lib)
    endif()
    if(TARGET quantized_kernels)
        executorch_target_link_options_shared_lib(quantized_kernels)
        target_link_libraries(app PRIVATE quantized_kernels)
    endif()
endif()
