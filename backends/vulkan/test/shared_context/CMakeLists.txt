# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(Threads REQUIRED)
add_executable(
  vulkan_shared_context_test SharedContextPolicyTest.cpp
                             SharedContextAdapterTest.cpp
)
target_include_directories(
  vulkan_shared_context_test PRIVATE ${COMMON_INCLUDES}
)
target_compile_options(vulkan_shared_context_test PRIVATE ${VULKAN_CXX_FLAGS})
target_link_libraries(
  vulkan_shared_context_test
  PRIVATE vulkan_backend executorch_vulkan_shared_runtime Threads::Threads
)
if(TARGET GTest::gtest_main)
  target_link_libraries(vulkan_shared_context_test PRIVATE GTest::gtest_main)
else()
  target_link_libraries(vulkan_shared_context_test PRIVATE gtest gtest_main)
endif()
set_property(TARGET vulkan_shared_context_test PROPERTY CXX_STANDARD 17)
add_test(NAME vulkan_shared_context_test COMMAND vulkan_shared_context_test)
set_tests_properties(vulkan_shared_context_test PROPERTIES TIMEOUT 180)
