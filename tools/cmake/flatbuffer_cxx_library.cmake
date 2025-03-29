# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/common.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_library.cmake)

function(flatbuffer_cxx_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SCHEMAS;DEPS"
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  elseif(NOT ARG_SCHEMAS)
    message(FATAL_ERROR "SCHEMAS is required")
  endif()

  # Output to same name as the target for consistency.
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR})

  set(generated_headers)
  foreach(schema ${ARG_SCHEMAS})
    get_filename_component(input_basename ${schema} NAME_WE)
    set(output_basename "${input_basename}_generated.h")
    set(output_destination "${output_dir}/${output_basename}")
    list(APPEND generated_headers ${output_destination})

    add_custom_command(
      OUTPUT ${output_destination}
      COMMAND flatc
                --cpp
                --cpp-std c++11
                --gen-mutable
                --scoped-enums
                -o ${output_dir}
                -c ${schema}
      DEPENDS flatc ${schema}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating ${schema} -> ${output_destination}"
    )
  endforeach()

  set_target_name(${ARG_NAME} TARGET_NAME)
  set(generator_name "${TARGET_NAME}_flatbuffer_generator")
  add_custom_target(${generator_name} ALL DEPENDS ${generated_headers})

  cxx_library(
    NAME ${ARG_NAME}
    EXTERNAL_HEADER_DIRS ${output_dir}/..
    PREPROCESSOR_DEFS "FLATBUFFERS_MAX_ALIGNMENT=1024"
    DEPS flatbuffers ${ARG_DEPS}
  )

  add_dependencies(${TARGET_NAME} ${generator_name})

  if(ARG_DEPS)
    add_dependencies(${TARGET_NAME} ${ARG_DEPS})
  endif()
endfunction()
