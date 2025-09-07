#[[
/*
* Copyright (c) 2024 MediaTek Inc.
*
* Licensed under the BSD License (the "License"); you may not use this file
* except in compliance with the License. See the license file in the root
* directory of this source tree for more details.
*/
]]

# Let include directory as "executorch/..."
set(_common_include_directories ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(EXECUTORCH_BUILD_NEURON_BUFFER_ALLOCATOR)
  message(STATUS "Neuron buffer allocator Build is enabled")
  add_subdirectory(runtime/proprietary)
endif()

include_directories(BEFORE ${_common_include_directories})

# shortcut include directory for neuron headers
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)

# targets
add_library(neuron_backend SHARED)
target_compile_options(neuron_backend PRIVATE "-frtti" "-fexceptions")
target_link_libraries(
  neuron_backend PRIVATE executorch_core portable_ops_lib portable_kernels
                         android log
)
target_sources(
  neuron_backend
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronBackend.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronBufferAllocator.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronExecutor.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronLog.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/api/APUWareUtilsLib.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/runtime/include/api/NeuronAdapterShim.h>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime/NeuronBackend.cpp
          ${CMAKE_CURRENT_LIST_DIR}/runtime/NeuronExecutor.cpp
          ${CMAKE_CURRENT_LIST_DIR}/runtime/NeuronBufferAllocator.cpp
)
executorch_target_link_options_shared_lib(neuron_backend)

install(
  TARGETS neuron_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
