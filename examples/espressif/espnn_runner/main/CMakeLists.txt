# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

set(runner_sources "main.cpp" "int8_suite.cpp" "bundle_data.S")
if(CONFIG_ESPNN_EXECUTORCH_DELEGATE)
  list(APPEND runner_sources "pte_suite.cpp"
       "${CMAKE_CURRENT_LIST_DIR}/../../executor_runner/esp_pal.cpp"
  )
endif()

idf_component_register(
  SRCS
  ${runner_sources}
  INCLUDE_DIRS
  "."
  REQUIRES
  espnn_runtime
  esp_timer
  heap
  esp_rom
  spi_flash
)

idf_build_get_property(project_dir PROJECT_DIR)
if(IS_ABSOLUTE "${CONFIG_ESPNN_RUNNER_BUNDLE}")
  set(bundle "${CONFIG_ESPNN_RUNNER_BUNDLE}")
else()
  set(bundle "${project_dir}/${CONFIG_ESPNN_RUNNER_BUNDLE}")
endif()
if(NOT EXISTS "${bundle}")
  message(
    FATAL_ERROR
      "Test bundle ${bundle} not found. Create one with "
      "examples/espressif/aot_espnn_compiler.py or examples/espressif/scripts/make_*.py "
      "and set CONFIG_ESPNN_RUNNER_BUNDLE."
  )
endif()
set_source_files_properties(
  "bundle_data.S"
  PROPERTIES COMPILE_DEFINITIONS "ESPNN_BUNDLE_FILE=\"${bundle}\""
             OBJECT_DEPENDS "${bundle}"
)
