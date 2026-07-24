# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Neutral, tensor-free KV-cache core shared across backends (registry,
# bookkeeping, config). No backend or tensor dependency. Sources are declared in
# build_variables.bzl (EXTENSION_LLM_CACHE_SRCS) so buck and cmake stay in sync.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

list(TRANSFORM _extension_llm_cache__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_llm_cache ${_extension_llm_cache__srcs})
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
