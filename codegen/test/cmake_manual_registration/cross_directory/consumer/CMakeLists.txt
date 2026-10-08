# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(CROSS_OPERATOR_MANUAL STREQUAL "OFF")
  gen_operators_lib(LIB_NAME cross_dir_ops SHARED)
else()
  gen_operators_lib(LIB_NAME cross_dir_ops MANUAL_REGISTRATION SHARED)
endif()
target_sources(
  cross_dir_ops PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../empty_runtime_stub.cpp
)
target_include_directories(cross_dir_ops PRIVATE ${SOURCE_INCLUDE_ROOT} ${TORCH_INCLUDE_ROOT})
target_compile_definitions(cross_dir_ops PRIVATE ET_LOG_ENABLED=0)
set_target_properties(cross_dir_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
add_executable(cross_directory_consumer cross_directory_consumer.cpp)
target_include_directories(
  cross_directory_consumer PRIVATE ${SOURCE_INCLUDE_ROOT} ${TORCH_INCLUDE_ROOT}
)
target_link_libraries(cross_directory_consumer PRIVATE cross_dir_ops)
