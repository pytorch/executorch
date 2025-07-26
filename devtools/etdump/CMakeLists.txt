# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_schema_files etdump_schema_flatcc.fbs scalar_type.fbs)

set(_schema_outputs)
foreach(schema_file ${_schema_files})
  list(APPEND _etdump_schema__srcs "${CMAKE_CURRENT_SOURCE_DIR}/${schema_file}")

  string(REGEX REPLACE "[.]fbs$" "_reader.h" generated_reader "${schema_file}")
  list(APPEND _schema_outputs
       "${DEVTOOLS_INCLUDE_DIR}/executorch/devtools/etdump/${generated_reader}"
  )

  string(REGEX REPLACE "[.]fbs$" "_builder.h" generated_builder
                       "${schema_file}"
  )
  list(APPEND _schema_outputs
       "${DEVTOOLS_INCLUDE_DIR}/executorch/devtools/etdump/${generated_builder}"
  )
endforeach()

file(MAKE_DIRECTORY
     ${DEVTOOLS_INCLUDE_DIR_NO_BUILD_INTERFACE}/executorch/devtools/etdump
)
add_custom_command(
  OUTPUT ${_schema_outputs}
  COMMAND
    # Note that the flatcc project actually writes its outputs into the source
    # tree instead of under the binary directory, and there's no way to change
    # that behavior.
    flatcc_cli -cwr -o
    ${DEVTOOLS_INCLUDE_DIR_NO_BUILD_INTERFACE}/executorch/devtools/etdump
    ${_etdump_schema__srcs}
  DEPENDS flatcc_cli ${_etdump_schema__srcs}
  COMMENT "Generating etdump headers"
)

add_library(
  etdump
  ${_schema_outputs}
  ${CMAKE_CURRENT_SOURCE_DIR}/etdump_flatcc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/emitter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/buffer_data_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/buffer_data_sink.h
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/file_data_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/file_data_sink.h
)
target_link_libraries(
  etdump
  PUBLIC flatccrt
  PRIVATE executorch
)
target_include_directories(
  etdump
  PUBLIC ${DEVTOOLS_INCLUDE_DIR}
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third-party/flatcc/include>
)

install(
  TARGETS etdump flatccrt
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
