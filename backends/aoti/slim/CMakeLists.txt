# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

# SlimTensor library for ExecuTorch CUDA backend A lightweight tensor
# implementation for AOTI (Ahead-of-Time Inference)

# C10 core headers
set(SLIM_C10_HEADERS
    c10/core/Device.h c10/core/DeviceType.h c10/Contiguity.h c10/MemoryFormat.h
    c10/SizesAndStrides.h c10/WrapDimMinimal.h
)

# Utility headers
set(SLIM_UTIL_HEADERS util/SharedPtr.h util/SizeUtil.h util/type_convert.h)

# Core SlimTensor headers
set(SLIM_CORE_HEADERS core/SlimTensor.h core/SlimTensorResize-incl.h
                      core/SlimTensorView-incl.h core/Storage.h
)

# Factory headers
set(SLIM_FACTORY_HEADERS factory/Empty.h factory/Factory.h factory/FromBlob.h
                         factory/FromScalar.h factory/Pad.h
)

# CUDA headers
set(SLIM_CUDA_HEADERS cuda/Exception.h cuda/Guard.h)

# All headers combined
set(SLIM_TENSOR_HEADERS
    ${SLIM_C10_HEADERS} ${SLIM_UTIL_HEADERS} ${SLIM_CORE_HEADERS}
    ${SLIM_FACTORY_HEADERS} ${SLIM_CUDA_HEADERS}
)

# Header-only interface library for SlimTensor
add_library(slim_tensor INTERFACE)
target_include_directories(
  slim_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
)

# Link to ExecuTorch dependencies
target_link_libraries(
  slim_tensor INTERFACE executorch_core extension_data_loader
)

# CUDA support (if available)
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(slim_tensor INTERFACE CUDA::cudart)
endif()

# Installation
install(FILES ${SLIM_C10_HEADERS}
        DESTINATION include/executorch/backends/aoti/slim/c10/core
)
install(FILES c10/Contiguity.h c10/MemoryFormat.h c10/SizesAndStrides.h
              c10/WrapDimMinimal.h
        DESTINATION include/executorch/backends/aoti/slim/c10
)
install(FILES ${SLIM_UTIL_HEADERS}
        DESTINATION include/executorch/backends/aoti/slim/util
)
install(FILES ${SLIM_CORE_HEADERS}
        DESTINATION include/executorch/backends/aoti/slim/core
)
install(FILES ${SLIM_FACTORY_HEADERS}
        DESTINATION include/executorch/backends/aoti/slim/factory
)
install(FILES ${SLIM_CUDA_HEADERS}
        DESTINATION include/executorch/backends/aoti/slim/cuda
)

# Tests (if building tests)
if(EXECUTORCH_BUILD_TESTS)
  enable_testing()

  # Basic SlimTensor tests
  add_executable(test_slim_tensor_basic tests/test_slim_tensor_basic.cpp)
  target_link_libraries(
    test_slim_tensor_basic PRIVATE slim_tensor gtest gtest_main
  )
  add_test(NAME test_slim_tensor_basic COMMAND test_slim_tensor_basic)

  # Type conversion tests
  add_executable(test_type_convert tests/test_type_convert.cpp)
  target_link_libraries(test_type_convert PRIVATE slim_tensor gtest gtest_main)
  add_test(NAME test_type_convert COMMAND test_type_convert)

  # CUDA tests (if CUDA is enabled)
  if(EXECUTORCH_BUILD_CUDA)
    add_executable(test_slim_tensor_cuda tests/test_slim_tensor_cuda.cpp)
    target_link_libraries(
      test_slim_tensor_cuda PRIVATE slim_tensor gtest gtest_main CUDA::cudart
    )
    add_test(NAME test_slim_tensor_cuda COMMAND test_slim_tensor_cuda)
  endif()
endif()
