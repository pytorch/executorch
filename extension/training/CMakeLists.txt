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
  # Pybind library.
  set(_pybind_training_dep_libs ${TORCH_PYTHON_LIBRARY} etdump executorch util
                                torch extension_training
  )

  if(EXECUTORCH_BUILD_XNNPACK)
    # need to explicitly specify XNNPACK and xnnpack-microkernels-prod here
    # otherwise uses XNNPACK and microkernel-prod symbols from libtorch_cpu
    list(APPEND _pybind_training_dep_libs xnnpack_backend XNNPACK
         xnnpack-microkernels-prod
    )
  endif()

  # pybind training
  pybind11_add_module(
    _training_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/pybindings/_training_lib.cpp
  )

  target_include_directories(_training_lib PRIVATE ${TORCH_INCLUDE_DIRS})
  target_compile_options(
    _training_lib PUBLIC -Wno-deprecated-declarations -fPIC -frtti -fexceptions
  )
  target_link_libraries(_training_lib PRIVATE ${_pybind_training_dep_libs})

  install(TARGETS _training_lib
          LIBRARY DESTINATION executorch/extension/training/pybindings
  )
endif()

# Install libraries
install(
  TARGETS extension_training
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
