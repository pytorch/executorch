# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cpu ${CMAKE_CURRENT_BINARY_DIR}/cpu_runtime
)

add_library(native_cpu_engine CPUEngine.cpp CPUEngineExecutable.cpp)
add_library(executorch::native_cpu_engine ALIAS native_cpu_engine)
target_link_libraries(
  native_cpu_engine
  PUBLIC native_engine cpu_runtime
  PRIVATE cpu_reference cpu_xnnpack extension_threadpool
          optimized_native_cpu_ops_lib
)
target_compile_options(
  native_cpu_engine PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

install(
  TARGETS native_cpu_engine
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  FILES CPUEngine.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/runtime/cpu
)
