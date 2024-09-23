# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

# Let include directory as "executorch/..."
set(_common_include_directories ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
include_directories(BEFORE ${_common_include_directories})

# shortcut include directory for neuron headers
include_directories(
  BEFORE ${_common_include_directories}/backends/mediatek/runtime/include
)

add_library(llm_helper STATIC)
target_sources(
  llm_helper
  PRIVATE llm_helper/mask_builder.cpp llm_helper/rotary_embedding.cpp
          llm_helper/token_embedding.cpp
)

target_link_libraries(llm_helper PRIVATE executorch)
target_include_directories(
  llm_helper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} llm_helper/include
)
target_compile_options(llm_helper PRIVATE ${_common_compile_options})

add_library(mtk_llama_executor_lib STATIC)
target_link_libraries(
  mtk_llama_executor_lib
  PRIVATE ${_executor_runner_libs} ${NEURON_BUFFER_ALLOCATOR_LIB}
          neuron_backend log llm_helper
)
target_sources(
  mtk_llama_executor_lib
  INTERFACE MultiModelLoader.h ModelChunk.h LlamaModelChunk.h LlamaRuntime.h
  PRIVATE MultiModelLoader.cpp ModelChunk.cpp LlamaModelChunk.cpp
          LlamaRuntime.cpp
)
target_compile_options(mtk_llama_executor_lib PUBLIC ${_common_compile_options})
