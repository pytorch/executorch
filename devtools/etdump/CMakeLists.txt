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

# The profiler is reachable from both the Python extension and a standalone C++
# application, and each one linking it statically would keep its own tracing
# state. Build it shared for the wheel, where both are loaded into one process,
# and keep it static everywhere else so no other build changes.
if(EXECUTORCH_BUILD_SHARED)
  set(_etdump_library_type SHARED)
else()
  set(_etdump_library_type STATIC)
endif()

add_library(
  etdump
  ${_etdump_library_type}
  ${_schema_outputs}
  ${CMAKE_CURRENT_SOURCE_DIR}/etdump_flatcc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/emitter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/buffer_data_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/buffer_data_sink.h
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/file_data_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_sinks/file_data_sink.h
)
# As with bundled_program, avoid the whole-archive of the `executorch` target so
# the primitive operator registrations are not duplicated alongside the copy
# already inside libexecutorch.so.
if(EXECUTORCH_BUILD_SHARED)
  # Private, not public: this library bundles the flatbuffer runtime rather than
  # depending on it, and the wheel does not ship flatccrt, so exposing it would
  # name something a consumer cannot link. Scoped to the shared build so a build
  # that does not opt in keeps the parent's public dependency.
  target_link_libraries(etdump PRIVATE flatccrt executorch_shared)
  set_target_properties(etdump PROPERTIES OUTPUT_NAME executorch_etdump)
  executorch_target_soname_policy(etdump)
  # Ships beside libexecutorch.so in the wheel's lib/ directory, and needs it,
  # so it has to be able to find it from wherever the package is installed.
  executorch_target_shipped_runtime_path(etdump)
else()
  target_link_libraries(etdump PUBLIC flatccrt)
  target_link_libraries(etdump PRIVATE executorch)
endif()

target_include_directories(
  etdump
  PUBLIC ${DEVTOOLS_INCLUDE_DIR}
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third-party/flatcc/include>
)

install(
  TARGETS etdump flatccrt
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
