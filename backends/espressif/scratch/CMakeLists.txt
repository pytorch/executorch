# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(esp_nn_scratch LANGUAGES C CXX)
endif()

if(MSVC
   OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
   OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
)
  message(
    FATAL_ERROR
      "ESP-NN scratch bindings require GNU-style section flags; MSVC-style toolchains are unsupported."
  )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../esp_nn_sources.cmake)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT TARGET pybind11::module)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pybind11"
    "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  )
endif()
set(common_sources
    src/convolution/esp_nn_conv_ansi.c
    src/convolution/esp_nn_depthwise_conv_ansi.c
    src/softmax/esp_nn_softmax_ansi.c
    src/common/esp_nn_mean_ansi.c
    src/activation_functions/esp_nn_hard_swish_ansi.c
    src/logistic/esp_nn_logistic_ansi.c
    src/fully_connected/esp_nn_fully_connected_ansi.c
)
set(esp32s3_sources
    src/convolution/esp_nn_scratch_sizes_esp32s3.c
    src/common/esp_nn_mean_s8_esp32s3.c
    src/activation_functions/esp_nn_hard_swish_s8_esp32s3.c
)
set(riscv_pie_sources src/convolution/esp_nn_scratch_sizes_riscv_pie.c)
set(generic_sources
    src/convolution/esp_nn_conv_opt.c
    src/convolution/esp_nn_depthwise_conv_opt.c
    src/softmax/esp_nn_softmax_opt.c
)

add_custom_target(_esp_nn_scratch)
foreach(profile esp32s3 riscv_pie generic ansi)
  set(sources ${common_sources} ${${profile}_sources})
  list(TRANSFORM sources PREPEND "${espnn_source_dir}/")
  set(module "_esp_nn_scratch_${profile}")
  pybind11_add_module(${module} MODULE bindings.cpp ${sources})
  target_compile_features(${module} PRIVATE cxx_std_17)
  add_dependencies(_esp_nn_scratch ${module})
  target_include_directories(
    ${module}
    PRIVATE "${espnn_source_dir}/include" "${espnn_source_dir}/src/common"
            "${espnn_source_dir}/src/convolution"
  )
  target_compile_definitions(${module} PRIVATE ESP_NN_SCRATCH_MODULE=${module})
  if(NOT profile STREQUAL "ansi")
    target_compile_definitions(${module} PRIVATE CONFIG_NN_OPTIMIZED=1)
  endif()
  if(profile STREQUAL "esp32s3")
    target_compile_definitions(${module} PRIVATE CONFIG_IDF_TARGET_ESP32S3=1)
  elseif(profile STREQUAL "riscv_pie")
    target_compile_definitions(${module} PRIVATE CONFIG_IDF_TARGET_ESP32P4=1)
  endif()

  # Some getters share a source file with kernels; retain only the sizing code.
  set_target_properties(${module} PROPERTIES C_VISIBILITY_PRESET hidden)
  target_compile_options(
    ${module} PRIVATE -ffunction-sections -fdata-sections
                      "$<$<COMPILE_LANGUAGE:CXX>:-fexceptions;-frtti>"
  )
  if(APPLE)
    target_link_options(${module} PRIVATE -Wl,-dead_strip)
  else()
    target_link_options(${module} PRIVATE -Wl,--gc-sections)
  endif()
endforeach()
