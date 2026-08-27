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

file(GLOB OBJC_SOURCES ExecuTorchETDump/Exported/*.m
     ExecuTorchETDump/Exported/*.mm
)

target_sources(extension_etdump_apple PRIVATE ${OBJC_SOURCES})

# The public header exposes ExecuTorchModule and so imports it as
# <ExecuTorch/ExecuTorchModule.h>, the path a consumer of the assembled
# framework uses. That layout does not exist during this build, so stage the
# core Exported headers under an ExecuTorch directory and put it on the include
# path, mirroring what the framework packaging produces.
set(_etdump_core_headers_dir ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${_etdump_core_headers_dir}/ExecuTorch)
file(GLOB _etdump_core_headers
     ${EXECUTORCH_ROOT}/extension/apple/ExecuTorch/Exported/*.h
)
file(COPY ${_etdump_core_headers}
     DESTINATION ${_etdump_core_headers_dir}/ExecuTorch
)

target_include_directories(
  extension_etdump_apple
  PUBLIC ExecuTorchETDump/Exported
  PRIVATE ${_etdump_core_headers_dir}
)

if(NOT TARGET etdump)
  message(
    FATAL_ERROR
      "EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE requires EXECUTORCH_BUILD_DEVTOOLS."
  )
endif()

find_library(FOUNDATION_FRAMEWORK Foundation)
target_link_libraries(
  extension_etdump_apple PRIVATE etdump extension_apple extension_module_static
                                 ${FOUNDATION_FRAMEWORK}
)

set_source_files_properties(
  ${OBJC_SOURCES} PROPERTIES COMPILE_FLAGS "-fobjc-arc" "-fno-exceptions"
                                                        "-fno-rtti"
)

set_target_properties(
  extension_etdump_apple
  PROPERTIES XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES
)
