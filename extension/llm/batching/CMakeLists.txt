# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Batched LLM serving: the scheduler that packs batches, the executor seam they
# run through, and the runner that carries whole generations across them. The
# scheduler and the executor seam are header-only and free of ExecuTorch runtime
# types; the runner owns a thread, so this is a static library rather than an
# INTERFACE target.
#
# extension_llm_batching_module is a separate target because it implements that
# seam against a program and a KV cache, and so carries the runtime types the
# seam itself is kept clear of.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_llm_batching runner.cpp metrics.cpp)
# std::optional, std::function, and std::future in the public headers.
target_compile_features(extension_llm_batching PUBLIC cxx_std_17)
target_include_directories(
  extension_llm_batching PUBLIC ${_common_include_directories}
)
target_compile_options(extension_llm_batching PUBLIC ${_common_compile_options})

# std::thread and friends need -pthread on glibc; a no-op where they do not.
find_package(Threads REQUIRED)
target_link_libraries(extension_llm_batching PUBLIC Threads::Threads)

add_library(extension_llm_batching_module module_executor.cpp)
target_link_libraries(
  extension_llm_batching_module
  PUBLIC extension_llm_batching extension_llm_cache extension_module
         extension_tensor
  PRIVATE extension_llm_sampler
)
target_include_directories(
  extension_llm_batching_module PUBLIC ${_common_include_directories}
)
target_compile_options(
  extension_llm_batching_module PUBLIC ${_common_compile_options}
)

install(
  TARGETS extension_llm_batching extension_llm_batching_module
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
