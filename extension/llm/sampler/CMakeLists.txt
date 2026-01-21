# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LLM sampler library with optional CUDA support
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# If the project is configured to build with CUDA support, build the CUDA
# sampler library.
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Define CUDA sampler library
    add_library(extension_llm_sampler STATIC argmax.cu)
    target_include_directories(
      extension_llm_sampler
      PUBLIC ${EXECUTORCH_ROOT}
             ${EXECUTORCH_ROOT}/..
             ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(extension_llm_sampler PUBLIC CUDA_AVAILABLE)
    target_link_libraries(extension_llm_sampler PUBLIC executorch_core
                                                            CUDA::cudart
    )
    set_target_properties(
      extension_llm_sampler
      PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON
    )

    message(
      STATUS "CUDAToolkit found; building extension_llm_sampler_cuda library"
    )

    install(
      TARGETS extension_llm_sampler
      EXPORT ExecuTorchTargets
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  else()
    message(
      STATUS
        "CUDA requested (EXECUTORCH_BUILD_CUDA=ON) but no CUDA runtime found"
    )
  endif()
endif()

# Install header files
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/sampler
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "test" EXCLUDE
)

