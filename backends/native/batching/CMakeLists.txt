# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The batching executor for native-backend packages: the native counterpart of
# extension_llm_batching_module. Added from backends/native after
# extension/llm/batching, whose executor seam it implements.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(native_batching native_executor.cpp)
target_include_directories(
  native_batching PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_options(
  native_batching PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                          $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)
target_link_libraries(
  native_batching
  PUBLIC native_engine extension_llm_batching extension_llm_batching_util
         extension_llm_cache extension_tensor
)

install(
  TARGETS native_batching
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  FILES native_executor.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/batching
)

if(BUILD_TESTING AND TARGET native_ggml_engine)
  include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

  # Skips unless ET_NATIVE_BATCHING_PTN names the output of
  # test/export_toy_llm.py.
  et_cxx_test(
    native_batching_test SOURCES test/test_native_executor.cpp EXTRA_LIBS
    native_batching native_ggml_engine
  )
  target_compile_options(
    native_batching_test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endif()
