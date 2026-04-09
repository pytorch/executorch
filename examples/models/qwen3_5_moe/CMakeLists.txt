# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(link_libraries
  executorch
  gflags
  optimized_native_cpu_ops_lib
  cpublas
  eigen_blas
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
  aoti_cuda_backend
  tokenizers::tokenizers
)

find_package(CUDAToolkit REQUIRED)

add_executable(qwen3_5_moe_runner main.cpp)
target_include_directories(
  qwen3_5_moe_runner PUBLIC ${EXECUTORCH_ROOT}/..
)
target_link_libraries(qwen3_5_moe_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(qwen3_5_moe_runner)
  target_link_options(qwen3_5_moe_runner PRIVATE "LINKER:-s")
endif()
