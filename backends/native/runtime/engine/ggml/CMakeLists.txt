# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(ggml 0.24 CONFIG REQUIRED)
if(NOT TARGET ggml::ggml-cpu)
  message(
    FATAL_ERROR
      "The native GGML engine requires a GGML package built with GGML_CPU=ON and GGML_BACKEND_DL=OFF"
  )
endif()

add_library(
  native_ggml_engine
  GgmlEngine.cpp
  GgmlExecutor.cpp
  GgmlGraphBuilder.cpp
  GgmlHostInts.cpp
  GgmlLowerAttention.cpp
  GgmlLowerConv.cpp
  GgmlLowerElementwise.cpp
  GgmlLowerLinear.cpp
  GgmlLowerShape.cpp
  GgmlTypes.cpp
)
add_library(executorch::native_ggml_engine ALIAS native_ggml_engine)
target_link_libraries(
  native_ggml_engine
  PUBLIC native_engine
  PRIVATE ggml::ggml
)
target_compile_options(
  native_ggml_engine PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                             $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

install(
  TARGETS native_ggml_engine
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  FILES GgmlEngine.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/runtime/engine/ggml
)
