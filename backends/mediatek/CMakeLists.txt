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
set(NEURON_BUFFER_ALLOCATOR_LIB
    ""
    CACHE PATH "Path to Neuron Buffer Allocator library"
)
message(
  STATUS "Looking for neuron_buffer_allocator in ${NEURON_BUFFER_ALLOCATOR_LIB}"
)

include_directories(BEFORE ${_common_include_directories})

# shortcut include directory for neuron headers
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)

# targets
add_library(neuron_backend SHARED)
target_link_libraries(neuron_backend
    PRIVATE
    executorch_no_prim_ops
    portable_ops_lib
    android
    log
    ${NEURON_BUFFER_ALLOCATOR_LIB}
)
target_sources(
  neuron_backend
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronBackend.h
            ${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronBufferAllocator.h
            ${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronExecutor.h
            ${CMAKE_CURRENT_LIST_DIR}/runtime/include/NeuronLog.h
            ${CMAKE_CURRENT_LIST_DIR}/runtime/include/api/APUWareUtilsLib.h
            ${CMAKE_CURRENT_LIST_DIR}/runtime/include/api/NeuronAdapterShim.h
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime/NeuronBackend.cpp
          ${CMAKE_CURRENT_LIST_DIR}/runtime/NeuronExecutor.cpp
)
target_link_options_shared_lib(neuron_backend)

install(TARGETS neuron_backend DESTINATION lib)
