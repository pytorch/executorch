# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

# ESP-IDF component providing the ExecuTorch ESP-NN delegate (EspNNBackend). It
# is meant to sit next to examples/espressif/executor_runner, which imports the
# cross-compiled ExecuTorch targets this component links against: build
# examples/espressif/project with -DET_ESPNN_BACKEND=ON.

idf_component_register(
  SRCS
  "src/xts3_graph.cpp"
  "src/int8_interpreter.cpp"
  "src/espnn_backend.cpp"
  INCLUDE_DIRS
  "include"
  REQUIRES
  esp_timer
  heap
  freertos
  PRIV_REQUIRES
  esp-nn
  WHOLE_ARCHIVE
)

get_filename_component(
  espnn_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
target_include_directories(
  ${COMPONENT_LIB}
  PRIVATE "${espnn_executorch_root}/.."
          "${espnn_executorch_root}/runtime/core/portable_type/c10"
)
target_compile_definitions(
  ${COMPONENT_LIB} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)
# Imported by executor_runner's find_package(executorch) as a global target.
target_link_libraries(${COMPONENT_LIB} PUBLIC executorch_core)
