# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_native_llama_cpp_dir
    ${EXECUTORCH_ROOT}/backends/native/third-party/llama.cpp
)
if(NOT EXISTS "${_native_llama_cpp_dir}/ggml/CMakeLists.txt"
   OR NOT EXISTS "${_native_llama_cpp_dir}/LICENSE"
)
  message(
    FATAL_ERROR
      "llama.cpp submodule is not initialized. Run: git submodule update --init backends/native/third-party/llama.cpp"
  )
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS OFF)
endif()
add_subdirectory(${_native_llama_cpp_dir}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
if(NOT TARGET ggml-cpu OR GGML_BACKEND_DL)
  message(
    FATAL_ERROR
      "The native GGML engine requires GGML_CPU=ON and GGML_BACKEND_DL=OFF"
  )
endif()

add_library(executorch_ggml INTERFACE IMPORTED)
target_link_libraries(executorch_ggml INTERFACE ggml)

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
  PRIVATE executorch_ggml
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
install(FILES ${_native_llama_cpp_dir}/LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/executorch/ggml
)
