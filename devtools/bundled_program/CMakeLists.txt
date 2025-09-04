# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_schema_files bundled_program_schema.fbs scalar_type.fbs)

set(_schema_outputs)
foreach(schema_file ${_schema_files})
  list(APPEND _bundled_program_schema__srcs
       "${CMAKE_CURRENT_SOURCE_DIR}/schema/${schema_file}"
  )

  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${schema_file}")
  list(
    APPEND
    _schema_outputs
    "${DEVTOOLS_INCLUDE_DIR}/executorch/devtools/bundled_program/schema/${generated}"
  )
endforeach()

file(
  MAKE_DIRECTORY
  ${DEVTOOLS_INCLUDE_DIR_NO_BUILD_INTERFACE}/executorch/devtools/bundled_program
)
add_custom_command(
  OUTPUT ${_schema_outputs}
  COMMAND
    flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums -o
    ${DEVTOOLS_INCLUDE_DIR}/executorch/devtools/bundled_program/schema
    ${_bundled_program_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/devtools
  DEPENDS flatc ${_bundled_program_schema__srcs}
  COMMENT "Generating bundled_program headers"
  VERBATIM
)

add_library(
  bundled_program ${_schema_outputs}
                  ${CMAKE_CURRENT_SOURCE_DIR}/bundled_program.cpp
)
target_link_libraries(bundled_program PUBLIC executorch)
target_include_directories(
  bundled_program
  PUBLIC
    ${DEVTOOLS_INCLUDE_DIR}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third-party/flatbuffers/include>
)

install(
  TARGETS bundled_program
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
