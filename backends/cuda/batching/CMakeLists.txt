# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Step slicing for the CUDA batching executor: cuts a batch of packed tokens
# into forwards the exported decode/prefill methods can run. Added from the root
# after extension/llm/batching.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

install(
  FILES step_plan.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/cuda/batching
)

if(BUILD_TESTING)
  include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

  et_cxx_test(test_step_plan SOURCES test/test_step_plan.cpp)
endif()
