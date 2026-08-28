# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_etdump_apple)

file(GLOB OBJC_SOURCES ExecuTorchDump/Exported/*.m ExecuTorchDump/Exported/*.mm)

target_sources(extension_etdump_apple PRIVATE ${OBJC_SOURCES})

# The public header exposes ExecuTorchModule and so imports it as
# <ExecuTorch/ExecuTorchModule.h>, the path a consumer of the assembled
# framework uses. That layout does not exist during this build, so stage the
# core Exported and Internal headers under one ExecuTorch directory and put it
# on the include path, mirroring what the framework packaging produces. Staging
# them together in one directory is what lets the Internal header's own quoted
# import of "ExecuTorchModule.h" find its sibling, and keeps the module header
# reachable by a single path so it is not seen as two conflicting definitions.
set(_etdump_core_headers_dir ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${_etdump_core_headers_dir}/ExecuTorch)
file(GLOB _etdump_core_headers
     ${EXECUTORCH_ROOT}/extension/apple/ExecuTorch/Exported/*.h
     ${EXECUTORCH_ROOT}/extension/apple/ExecuTorch/Internal/*.h
)
file(COPY ${_etdump_core_headers}
     DESTINATION ${_etdump_core_headers_dir}/ExecuTorch
)

target_include_directories(
  extension_etdump_apple
  PUBLIC ExecuTorchDump/Exported
  PRIVATE ExecuTorchDump/Internal
          ${_etdump_core_headers_dir}
          # For <executorch/...> includes reached through etdump_flatcc.h.
          ${EXECUTORCH_ROOT}/..
          # For the c10 portable types those headers pull in.
          ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# The wrapper pulls in the core tensor headers through etdump_flatcc.h, which
# reach c10's Macros.h. Without this define that header tries to include a
# generated cmake_macros.h the Apple build does not produce, so use the custom
# path the other Apple targets use.
target_compile_definitions(
  extension_etdump_apple PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)

if(NOT TARGET etdump)
  message(
    FATAL_ERROR
      "EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE requires the etdump target, "
      "which the root CMakeLists builds when this option is on. It is missing "
      "here, so the build was configured in a way that skipped it."
  )
endif()

if(NOT TARGET extension_apple)
  message(FATAL_ERROR "EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE requires "
                      "EXECUTORCH_BUILD_EXTENSION_APPLE."
  )
endif()

if(NOT TARGET extension_module_static)
  message(FATAL_ERROR "EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE requires "
                      "EXECUTORCH_BUILD_EXTENSION_MODULE."
  )
endif()

find_library(FOUNDATION_FRAMEWORK Foundation)
target_link_libraries(
  extension_etdump_apple PRIVATE etdump extension_apple extension_module_static
                                 ${FOUNDATION_FRAMEWORK}
)

set_source_files_properties(
  ${OBJC_SOURCES} PROPERTIES COMPILE_FLAGS
                             "-fobjc-arc -fno-exceptions -fno-rtti"
)

set_target_properties(
  extension_etdump_apple
  PROPERTIES XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES
)

# The Apple framework build merges each static library straight from the target
# output directory, but flatccrt (the flatcc runtime the etdump framework
# bundles) is built into its own third-party/flatcc/lib directory, not beside
# this target. Copy it next to this library after it builds so the framework
# packaging finds it where it expects. TARGET_FILE_DIR resolves to the
# per-config directory the packaging reads, so this is correct under every
# generator.
add_custom_command(
  TARGET extension_etdump_apple
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:flatccrt>
          $<TARGET_FILE_DIR:extension_etdump_apple>/libflatccrt.a
  VERBATIM
)
