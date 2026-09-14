# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(
  shared_gpu_runtime_test
  ${CMAKE_CURRENT_SOURCE_DIR}/SharedGpuContextRegistryTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SharedGpuRuntimeConfigTest.cpp
)

target_link_libraries(
  shared_gpu_runtime_test PRIVATE executorch_gpu_shared_runtime
                                  Threads::Threads
)

if(TARGET GTest::gtest_main)
  target_link_libraries(shared_gpu_runtime_test PRIVATE GTest::gtest_main)
else()
  target_link_libraries(shared_gpu_runtime_test PRIVATE gtest gtest_main)
endif()

add_test(NAME shared_gpu_runtime_test COMMAND shared_gpu_runtime_test)
