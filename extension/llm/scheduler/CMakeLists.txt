# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Step scheduler for batched LLM serving. Header-only and free of ExecuTorch
# runtime types, so it is an INTERFACE target: consumers get the include path
# and nothing to link.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_llm_scheduler INTERFACE)
# std::optional and std::variant in the public headers.
target_compile_features(extension_llm_scheduler INTERFACE cxx_std_17)
target_include_directories(
  extension_llm_scheduler INTERFACE ${_common_include_directories}
)
target_compile_options(
  extension_llm_scheduler INTERFACE ${_common_compile_options}
)

install(
  TARGETS extension_llm_scheduler
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/scheduler
  FILES_MATCHING
  PATTERN "*.h"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
