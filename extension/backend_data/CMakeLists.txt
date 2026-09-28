# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

add_library(
  extension_backend_data buffer_data_writer.cpp file_data_writer.cpp
                         initialize_and_save.cpp
)
target_link_libraries(
  extension_backend_data
  PUBLIC executorch_core
  PRIVATE program_schema
)
target_include_directories(
  extension_backend_data PUBLIC ${_common_include_directories}
)
target_compile_options(extension_backend_data PUBLIC ${_common_compile_options})

install(
  TARGETS extension_backend_data
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
