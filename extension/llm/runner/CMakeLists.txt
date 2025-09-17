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

add_library(extension_llm_runner STATIC ${_extension_llm_runner__srcs})

set(runner_deps executorch_core extension_module extension_tensor
                tokenizers::tokenizers
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
set_target_properties(
  extension_llm_runner PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  extension_llm_runner INTERFACE ${_common_include_directories}
)

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
# TODO: remove this once we create a proper CMake setup for sampler.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../sampler/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/sampler
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
                        portable_lib
  )

  # Set properties for the Python extension
  set_target_properties(
    _llm_runner
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET "hidden"
               INTERPROCEDURAL_OPTIMIZATION TRUE
  )

  # Add include directories
  target_include_directories(
    _llm_runner PRIVATE ${_common_include_directories} ${TORCH_INCLUDE_DIRS}
  )

  install(TARGETS _llm_runner
          LIBRARY DESTINATION executorch/extension/llm/runner
  )
endif()
