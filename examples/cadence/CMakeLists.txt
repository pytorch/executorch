# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set the minimum required version of CMake for this project.
cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set the project name.
project(cadence_executorch_example)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# Find prebuilt executorch lib
find_package(executorch CONFIG REQUIRED)

add_compile_options(
  -DSDK_DEBUGCONSOLE=1
  -DSERIAL_PORT_TYPE_UART=1
  -DDEBUG_CONSOLE_RX_ENABLE=0
  -DDEBUG
  -DCPU_MIMXRT685SFVKB_dsp
  -DMCUXPRESSO_SDK
  -g
  -O0
  -Wall
  -fsigned-char
  -Wno-missing-braces
  -fmessage-length=0
  -DPRINTF_FLOAT_ENABLE=1
)

if(NOT DEFINED NXP_SDK_ROOT_DIR)
  message(FATAL_ERROR "NXP_SDK_ROOT_DIR is not set")
endif()

# lint_cmake: -linelength
set(SOURCES
    ${NXP_SDK_ROOT_DIR}/components/lists/fsl_component_generic_list.c
    ${NXP_SDK_ROOT_DIR}/components/uart/fsl_adapter_usart.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_clock.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_common.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_common_dsp.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_flexcomm.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_gpio.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_mu.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_reset.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers/fsl_usart.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/system_MIMXRT685S_dsp.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/utilities/debug_console_lite/fsl_assert.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/utilities/debug_console_lite/fsl_debug_console.c
    ${NXP_SDK_ROOT_DIR}/boards/evkmimxrt685/dsp_examples/mu_polling/hifi4/board_hifi4.c
    ${NXP_SDK_ROOT_DIR}/boards/evkmimxrt685/dsp_examples/mu_polling/hifi4/pin_mux.c
    ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/utilities/str/fsl_str.c
)

add_library(dsp_mu_polling_libs STATIC ${SOURCES})

target_include_directories(
  dsp_mu_polling_libs
  PUBLIC ${NXP_SDK_ROOT_DIR}
         ${NXP_SDK_ROOT_DIR}/components/uart
         ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/drivers
         ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/periph
         ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/utilities/debug_console_lite
         ${NXP_SDK_ROOT_DIR}/components/lists
         ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S
         ${NXP_SDK_ROOT_DIR}/CMSIS/Core/Include
         ${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/utilities/str
         ${NXP_SDK_ROOT_DIR}/boards/evkmimxrt685/dsp_examples/mu_polling/hifi4
)

add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib
  ${CMAKE_CURRENT_BINARY_DIR}/backends/cadence/hifi/third-party/nnlib
)

add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/operators
  ${CMAKE_CURRENT_BINARY_DIR}/backends/cadence/hifi/operators
)
add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/kernels
  ${CMAKE_CURRENT_BINARY_DIR}/backends/cadence/hifi/kernels
)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). In order to fit the model into i.MXRT600 memory, we
# use selective kernel registration based on MODEL_PATH, and lik to
# cadence_ops_lib_selective.
gen_selected_ops(
  LIB_NAME "cadence_ops_lib_selective" OPS_FROM_MODEL "${MODEL_PATH}"
)
generate_bindings_for_kernels(
  LIB_NAME "cadence_ops_lib_selective" OPS_SCHEMA_YAML FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/backends/cadence/aot/functions_hifi.yaml
)
message("Generated files: ${gen_command_sources}")

gen_operators_lib(
  LIB_NAME "cadence_ops_lib_selective" KERNEL_LIBS custom_ops DEPS
  aten_ops_cadence
)

# Generate the model header file
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/model_pte.h
  COMMAND
    ${PYTHON_EXECUTABLE} ${EXECUTORCH_ROOT}/backends/cadence/utils/gen_header.py
    --model_path ${MODEL_PATH} --header_output_path ${CMAKE_BINARY_DIR}
  COMMENT "Converting .pte model to header file..."
  DEPENDS ${EXECUTORCH_ROOT}/backends/cadence/utils/gen_header.py
)

add_custom_target(gen_model_header DEPENDS ${CMAKE_BINARY_DIR}/model_pte.h)

add_executable(
  cadence_executorch_example
  ${EXECUTORCH_ROOT}/backends/cadence/executor_runner.cpp
)
add_dependencies(cadence_executorch_example gen_model_header)

# lint_cmake: -linelength
target_include_directories(
  cadence_executorch_example PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                                    ${_common_include_directories}
)

target_link_options(
  cadence_executorch_example PRIVATE
  -mlsp=${NXP_SDK_ROOT_DIR}/devices/MIMXRT685S/xtensa/min-rt
  -Wl,--orphan-handling=place
)
target_link_libraries(
  cadence_executorch_example dsp_mu_polling_libs cadence_ops_lib_selective
  extension_runner_util executorch
)

add_custom_command(
  TARGET cadence_executorch_example
  POST_BUILD
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${EXECUTORCH_ROOT}/backends/cadence/utils/post_compilation.py
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.elf ${CMAKE_BINARY_DIR}
  COMMENT
    "Generating .bin files that can be used to flash the DSP with. Copy over
    the dsp_text_release.bin and dsp_data_release.bin that are generated into
    your NXP MCUXpresso IDE workspace and flash the DSP with these binaries."
    DEPENDS
    ${EXECUTORCH_ROOT}/backends/cadence/utils/post_compilation.py
)
