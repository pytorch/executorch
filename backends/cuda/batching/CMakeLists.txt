# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The batching executor for programs the CUDA backend compiled: the CUDA
# counterpart of extension_llm_batching_module. Added from the root after
# extension/llm/batching, whose executor seam it implements.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(cuda_batching cuda_executor.cpp)
target_include_directories(cuda_batching PUBLIC ${_common_include_directories})
target_compile_options(cuda_batching PUBLIC ${_common_compile_options})
target_compile_definitions(cuda_batching PRIVATE CUDA_AVAILABLE=1)
target_link_libraries(
  cuda_batching
  PUBLIC aoti_cuda_backend extension_llm_batching extension_llm_cache
         extension_module extension_tensor
  PRIVATE extension_llm_sampler
)

install(
  TARGETS cuda_batching
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  FILES cuda_executor.h step_plan.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/cuda/batching
)

if(BUILD_TESTING)
  include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

  et_cxx_test(test_step_plan SOURCES test/test_step_plan.cpp)

  # Skips unless ET_CUDA_BATCHING_TOY_DIR names the output of
  # test/export_toy_decoder.py. The program's host/device copies are portable
  # et_copy kernels, registered by the CPU ops library.
  et_cxx_test(
    test_cuda_executor SOURCES test/test_cuda_executor.cpp EXTRA_LIBS
    cuda_batching optimized_native_cpu_ops_lib
  )
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
endif()
