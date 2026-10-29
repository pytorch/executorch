# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

get_filename_component(
  camera_et_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE
)

# The ExecuTorch platform layer is shared with the executor runner example.
# WHOLE_ARCHIVE keeps it over the runtime's weak defaults.
idf_component_register(
  SRCS
  "main.cpp"
  "camera_stream.cpp"
  "et_model.cpp"
  "model_data.S"
  "${camera_et_root}/examples/espressif/executor_runner/esp_pal.cpp"
  INCLUDE_DIRS
  "."
  REQUIRES
  espnn_runtime
  esp32-camera
  esp_jpeg
  esp_timer
  esp_rom
  esp_driver_usb_serial_jtag
  WHOLE_ARCHIVE
)

set(ET_BUILD_DIR_PATH
    "${camera_et_root}/cmake-out-esp"
    CACHE PATH "ExecuTorch cross-compiled for this target and installed"
)
# GLOBAL so that the project-level link step and espnn_runtime see the imported
# targets.
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
find_package(
  executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
)
target_include_directories(
  ${COMPONENT_LIB} PRIVATE "${camera_et_root}/.."
                           "${camera_et_root}/runtime/core/portable_type/c10"
)
target_compile_definitions(
  ${COMPONENT_LIB} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)
set(camera_et_libs executorch executorch_selected_kernels)
if(TARGET quantized_ops_lib)
  list(APPEND camera_et_libs quantized_ops_lib)
endif()
target_link_libraries(${COMPONENT_LIB} PRIVATE ${camera_et_libs})

idf_build_get_property(project_dir PROJECT_DIR)
if(IS_ABSOLUTE "${CONFIG_CAMERA_STREAM_MODEL}")
  set(model "${CONFIG_CAMERA_STREAM_MODEL}")
else()
  set(model "${project_dir}/${CONFIG_CAMERA_STREAM_MODEL}")
endif()
if(NOT EXISTS "${model}")
  message(
    FATAL_ERROR
      "Camera model ${model} not found. Export one with "
      "examples/espressif/camera_stream/export_model.py and set "
      "CONFIG_CAMERA_STREAM_MODEL."
  )
endif()
set_source_files_properties(
  "model_data.S" PROPERTIES COMPILE_DEFINITIONS "CAMERA_MODEL_FILE=\"${model}\""
                            OBJECT_DEPENDS "${model}"
)
