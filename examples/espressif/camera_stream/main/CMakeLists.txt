# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

idf_component_register(
  SRCS
  "main.cpp"
  "camera_stream.cpp"
  "model_data.S"
  INCLUDE_DIRS
  "."
  REQUIRES
  espnn_runtime
  esp32-camera
  esp_jpeg
  esp_timer
  esp_rom
  esp_driver_usb_serial_jtag
)

idf_build_get_property(project_dir PROJECT_DIR)
if(IS_ABSOLUTE "${CONFIG_CAMERA_STREAM_MODEL}")
  set(model "${CONFIG_CAMERA_STREAM_MODEL}")
else()
  set(model "${project_dir}/${CONFIG_CAMERA_STREAM_MODEL}")
endif()
if(NOT EXISTS "${model}")
  message(
    FATAL_ERROR
      "Camera model ${model} not found. Compile one with "
      "examples/espressif/aot_espnn_compiler.py --mode standalone and set "
      "CONFIG_CAMERA_STREAM_MODEL."
  )
endif()
set_source_files_properties(
  "model_data.S" PROPERTIES COMPILE_DEFINITIONS "CAMERA_MODEL_FILE=\"${model}\""
                            OBJECT_DEPENDS "${model}"
)
