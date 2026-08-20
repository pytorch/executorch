# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install / rendezvous registry for the off-graph KV cache, shared across
# backends. The neutral cache handle (cache.h) is ET-independent, so it is
# usable outside an ET runner; the delegate-specific registry
# (cache_registry.cpp) uses ExecuTorch Error/Result, hence the executorch_core
# link.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_llm_cache cache_registry.cpp)
target_link_libraries(extension_llm_cache executorch_core)
target_include_directories(
  extension_llm_cache PUBLIC ${_common_include_directories}
)
target_compile_options(extension_llm_cache PUBLIC ${_common_compile_options})

install(
  TARGETS extension_llm_cache
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/cache
  FILES_MATCHING
  PATTERN "*.h"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
