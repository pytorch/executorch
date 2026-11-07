# Copyright 2026 Arm Limited and/or its affiliates.
# Licensed under the BSD-style license in the root LICENSE file.

if(NOT EXECUTORCH_BUILD_VULKAN OR NOT EXECUTORCH_BUILD_VGF)
  message(FATAL_ERROR "The mixed example requires both Vulkan and VGF backends")
endif()
if(NOT EXECUTORCH_BUILD_EXTENSION_MODULE OR NOT
                                            EXECUTORCH_BUILD_EXTENSION_TENSOR
)
  message(
    FATAL_ERROR
      "Enable EXECUTORCH_BUILD_EXTENSION_MODULE and EXECUTORCH_BUILD_EXTENSION_TENSOR"
  )
endif()

add_executable(vgf_vulkan_shared_runner runner.cpp)
target_compile_features(vgf_vulkan_shared_runner PRIVATE cxx_std_17)
target_compile_options(
  vgf_vulkan_shared_runner
  PRIVATE "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>"
          "$<$<CXX_COMPILER_ID:MSVC>:/EHsc>"
)
target_link_libraries(
  vgf_vulkan_shared_runner
  PRIVATE extension_module extension_tensor vgf_backend vulkan_backend
          executorch_vulkan_shared_runtime
)
# The exporter rejects CPU KernelCall instructions; no fallback kernels needed.
set_target_properties(
  vgf_vulkan_shared_runner PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH
                                                            "$ORIGIN"
)
