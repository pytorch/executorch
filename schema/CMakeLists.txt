# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Flatbuffer schema header lib. Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

# The include directory that will contain the generated schema headers.
set(_program_schema__include_dir "${CMAKE_CURRENT_BINARY_DIR}/schema/include")

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Paths to headers generated from the .fbs files.
set(_program_schema__srcs program.fbs scalar_type.fbs)

set(_program_schema__outputs)
foreach(fbs_file ${_program_schema__srcs})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(APPEND _program_schema__outputs
       "${_program_schema__include_dir}/executorch/${generated}")
endforeach()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_program_schema__outputs}
  COMMAND
    flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums
    # Add a subdirectory to the include dir so the files can be included as
    # <executorch/schema/x_generated.h>
    -o "${_program_schema__include_dir}/executorch/schema"
    ${_program_schema__srcs}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS flatc ${_program_schema__srcs}
  COMMENT "Generating program_schema headers"
  VERBATIM)

add_library(program_schema INTERFACE ${_program_schema__outputs})
set_target_properties(program_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  program_schema INTERFACE ${_program_schema__include_dir}
                           ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)
