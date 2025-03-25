# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE
# file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT

file(GLOB source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
get_filename_component(tool_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_executable(${tool_name} ${source_files})
target_link_libraries(${tool_name} PRIVATE tokenizers)
target_include_directories(${tool_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/include/pytorch/tokenizers
)
