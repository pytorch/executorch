# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT TARGET vulkan_graph_runtime)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/backends/vulkan
    ${CMAKE_CURRENT_BINARY_DIR}/vulkan_runtime
  )
endif()

file(GLOB _native_vulkan_passes_srcs CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/passes/*.cpp
)
add_library(native_vulkan_passes ${_native_vulkan_passes_srcs})
target_link_libraries(
  native_vulkan_passes
  PUBLIC native_graph
  PRIVATE native_reference_gguf_kquant
)

add_library(
  native_vulkan_engine VulkanConstantMaterializationTracker.cpp
                       VulkanEngine.cpp VulkanEngineExecutable.cpp
)
add_library(executorch::native_vulkan_engine ALIAS native_vulkan_engine)
target_link_libraries(
  native_vulkan_engine
  PUBLIC native_engine
  PRIVATE native_reference_gguf_kquant native_vulkan_passes
          vulkan_graph_runtime
)

foreach(target native_vulkan_passes native_vulkan_engine)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endforeach()

install(
  TARGETS native_vulkan_passes native_vulkan_engine
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  FILES VulkanEngine.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/runtime/vulkan
)
install(
  DIRECTORY passes/
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/runtime/vulkan/passes
  FILES_MATCHING
  PATTERN "*.h"
)
