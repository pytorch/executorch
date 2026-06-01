# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Build llm runner lib.
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
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

# build llm runner library
list(TRANSFORM _extension_llm_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

# The Jinja chat formatter requires jinja2cpp. When jinja2cpp is not built (for
# example a runner build that does not add the chat_template subdirectory), drop
# the source so extension_llm_runner has no dependency on jinja2cpp at all.
if(NOT TARGET jinja2cpp)
  list(FILTER _extension_llm_runner__srcs EXCLUDE REGEX
       "extension/llm/runner/jinja_chat_formatter\\.cpp$"
  )
endif()

add_library(extension_llm_runner STATIC ${_extension_llm_runner__srcs})

# include sampler
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/sampler
  ${CMAKE_CURRENT_BINARY_DIR}/../sampler
)

set(runner_deps
    executorch_core extension_module extension_tensor extension_llm_sampler
    extension_memory_allocator tokenizers::tokenizers
)

# depend on arange_utils
if(NOT TARGET kernels_util_all_deps)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/kernels/portable/cpu/util
    ${CMAKE_CURRENT_BINARY_DIR}/kernels_util
  )
endif()
list(APPEND runner_deps kernels_util_all_deps)

target_link_libraries(extension_llm_runner PUBLIC ${runner_deps})
# jinja2cpp is a private, build-tree-only dependency of the Jinja chat
# formatter. It is a FetchContent static library that is not installed, so it
# must NOT leak into the installed ExecuTorchTargets export -- otherwise
# downstream find_package(executorch) consumers fail to link with `-ljinja2cpp`.
# BUILD_INTERFACE keeps the dependency local to this build tree.
if(TARGET jinja2cpp)
  target_link_libraries(
    extension_llm_runner PRIVATE $<BUILD_INTERFACE:jinja2cpp>
  )
  target_include_directories(
    extension_llm_runner
    PRIVATE $<TARGET_PROPERTY:jinja2cpp,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()
set_target_properties(
  extension_llm_runner PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  extension_llm_runner INTERFACE ${_common_include_directories}
)

# If the project is configured to build with CUDA support, try to find a CUDA
# runtime (prefer the CUDAToolkit package). If found, expose a compile-time
# macro so sources can conditionally compile CUDA-aware code.
if(EXECUTORCH_BUILD_CUDA)
  # Prefer the modern CMake CUDAToolkit module, fall back to searching for the
  # CUDA runtime library (cudart) if the package isn't available.
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_compile_definitions(extension_llm_runner PUBLIC CUDA_AVAILABLE)
    target_link_libraries(extension_llm_runner PUBLIC CUDA::cudart)
    message(STATUS "CUDAToolkit found; defining CUDA_AVAILABLE")
  else()
    message(
      STATUS
        "CUDA requested (EXECUTORCH_BUILD_CUDA=ON) but no CUDA runtime found"
    )
  endif()
endif()

install(
  TARGETS extension_llm_runner
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/runner
  FILES_MATCHING
  PATTERN "*.h"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# Python bindings for MultimodalRunner
if(EXECUTORCH_BUILD_PYBIND)
  # Create the Python extension module for LLM runners
  pybind11_add_module(
    _llm_runner SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pybindings.cpp
  )

  find_package_torch()
  find_library(
    TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib"
  )
  # Link with the extension_llm_runner library and its dependencies
  target_link_libraries(
    _llm_runner PRIVATE extension_llm_runner tokenizers::tokenizers
                        portable_lib ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES}
  )

  set_target_properties(
    _llm_runner
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET "hidden"
               INTERPROCEDURAL_OPTIMIZATION TRUE
               CXX_STANDARD 20
  )
  if(APPLE)
    set(RPATH
        "@loader_path/../../pybindings;@loader_path/../../../../torch/lib"
    )
  else()
    set(RPATH "$ORIGIN/../../pybindings:$ORIGIN/../../../../torch/lib")
  endif()
  set_target_properties(
    _llm_runner PROPERTIES BUILD_RPATH "${RPATH}" INSTALL_RPATH "${RPATH}"
  )
  # Add include directories
  target_include_directories(
    _llm_runner PRIVATE ${_common_include_directories} ${TORCH_INCLUDE_DIRS}
  )

  install(TARGETS _llm_runner
          LIBRARY DESTINATION executorch/extension/llm/runner
  )
endif()
