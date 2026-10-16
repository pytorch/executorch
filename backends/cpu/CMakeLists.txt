# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(
  cpu_runtime
  runtime/KernelProvider.cpp
  runtime/CPUPlan.cpp
  runtime/op_helpers/Convolution.cpp
)
target_link_libraries(cpu_runtime PUBLIC native_runtime executorch_core)

add_library(cpu_reference runtime/providers/reference/ReferenceProvider.cpp)
target_link_libraries(cpu_reference PUBLIC cpu_runtime)

add_library(cpu_xnnpack runtime/providers/xnnpack/XNNPACKProvider.cpp)
target_link_libraries(
  cpu_xnnpack PUBLIC cpu_runtime PRIVATE XNNPACK extension_threadpool
)

foreach(target cpu_runtime cpu_reference cpu_xnnpack)
  add_library(executorch::${target} ALIAS ${target})
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endforeach()

install(
  TARGETS cpu_runtime cpu_reference cpu_xnnpack
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY runtime/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/cpu/runtime
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
)
