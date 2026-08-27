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

target_include_directories(
  extension_etdump_apple PUBLIC ExecuTorchETDump/Exported
)

if(NOT TARGET etdump)
  message(
    FATAL_ERROR
      "EXECUTORCH_BUILD_EXTENSION_ETDUMP_APPLE requires EXECUTORCH_BUILD_DEVTOOLS."
  )
endif()

find_library(FOUNDATION_FRAMEWORK Foundation)
target_link_libraries(
  extension_etdump_apple PRIVATE etdump extension_module_static
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
