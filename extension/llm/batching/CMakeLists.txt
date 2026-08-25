# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Batched LLM serving: the step scheduler, and the runner that drives batches
# through a model. Currently header-only and free of ExecuTorch runtime types,
# so it is an INTERFACE target; this becomes a real library once the runner
# lands with a .cpp.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_llm_batching INTERFACE)
# std::optional in the public headers.
target_compile_features(extension_llm_batching INTERFACE cxx_std_17)
target_include_directories(
  extension_llm_batching INTERFACE ${_common_include_directories}
)
target_compile_options(
  extension_llm_batching INTERFACE ${_common_compile_options}
)

install(
  TARGETS extension_llm_batching
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/batching
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
