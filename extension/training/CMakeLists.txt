# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

list(TRANSFORM _extension_training__srcs PREPEND "${EXECUTORCH_ROOT}/")

add_library(extension_training ${_extension_training__srcs})
target_include_directories(
  extension_training PUBLIC ${_common_include_directories}
)

target_compile_options(extension_training PUBLIC ${_common_compile_options})
target_link_libraries(
  extension_training
  executorch_core
  kernels_util_all_deps
  extension_data_loader
  extension_module_static
  extension_tensor
  extension_flat_tensor
)

list(TRANSFORM _train_xor__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_executable(train_xor ${_train_xor__srcs})
target_include_directories(train_xor PUBLIC ${_common_include_directories})
target_link_libraries(
  train_xor
  gflags
  executorch_core
  portable_ops_lib
  extension_tensor
  extension_training
  program_schema
)
target_compile_options(train_xor PUBLIC ${_common_compile_options})

if(EXECUTORCH_BUILD_PYBIND)
  # Pybind library. When the consolidated shared runtime is built, the runtime
  # is resolved from it rather than from the whole-archive-forcing static
  # `executorch` target, so this module shares the one backend registry.
  if(EXECUTORCH_BUILD_SHARED)
    set(_pybind_training_dep_libs ${TORCH_PYTHON_LIBRARY} etdump util torch
                                  extension_training
    )
  else()
    set(_pybind_training_dep_libs ${TORCH_PYTHON_LIBRARY} etdump executorch
                                  util torch extension_training
    )
  endif()

  if(EXECUTORCH_BUILD_XNNPACK)
    if(EXECUTORCH_BUILD_SHARED)
      # The delegate bundles XNNPACK and its microkernels, so naming them again
      # here would ask for a second copy of what that library already provides.
      list(APPEND _pybind_training_dep_libs xnnpack_backend)
    else()
      # need to explicitly specify XNNPACK and xnnpack-microkernels-prod here
      # otherwise uses XNNPACK and microkernel-prod symbols from libtorch_cpu
      list(APPEND _pybind_training_dep_libs xnnpack_backend XNNPACK
           xnnpack-microkernels-prod
      )
    endif()
  endif()

  pybind11_add_module(
    _training_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/pybindings/_training_lib.cpp
  )
  strip_python_lib(_training_lib)
  set_target_properties(_training_lib PROPERTIES CXX_STANDARD 20)

  target_include_directories(_training_lib PRIVATE ${TORCH_INCLUDE_DIRS})
  target_compile_options(
    _training_lib
    PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc
           /GR
           /wd4996>
           $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations
           -fPIC
           -frtti
           -fexceptions>
  )
  target_link_libraries(_training_lib PRIVATE ${_pybind_training_dep_libs})

  if(EXECUTORCH_BUILD_SHARED
     AND EXECUTORCH_BUILD_XNNPACK
     AND TARGET xnnpack_backend
  )
    # The delegate registers itself from a static initializer, so nothing in
    # this extension references a symbol from it and a normal link can drop it.
    # Keeping it named on the link line is what makes an XNNPACK-delegated
    # program usable from here.
    executorch_target_retain_shared_library(_training_lib xnnpack_backend)
  endif()
  executorch_target_link_shared_runtime(_training_lib)

  if(EXECUTORCH_BUILD_SHARED)
    # This module links Torch directly, and the only other entry reaching it is
    # the absolute build directory CMake adds, which does not exist anywhere
    # else, so the Torch path is recorded here rather than left implicit.
    if(APPLE)
      set(_training_torch_path "@loader_path/../../../../torch/lib")
    else()
      set(_training_torch_path "$ORIGIN/../../../../torch/lib")
    endif()
    set_target_properties(
      _training_lib PROPERTIES INSTALL_RPATH "${_training_torch_path}"
    )
    executorch_target_shared_runtime_path(
      _training_lib "extension/training/pybindings"
      "executorch/extension/training/pybindings"
    )
  endif()

  install(TARGETS _training_lib
          LIBRARY DESTINATION executorch/extension/training/pybindings
  )
endif()

# Install libraries
install(
  TARGETS extension_training
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

# Headers
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/training
  FILES_MATCHING
  PATTERN "*.h"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/optimizer
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/training
  FILES_MATCHING
  PATTERN "*.h"
)
