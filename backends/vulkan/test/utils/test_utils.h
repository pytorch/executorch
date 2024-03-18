/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <ATen/native/vulkan/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

using namespace at::native::vulkan;

#define CREATE_FLOAT_TEXTURE(sizes, allocate_memory) \
  vTensor(                                           \
      api::context(),                                \
      sizes,                                         \
      api::kFloat,                                   \
      api::StorageType::TEXTURE_3D,                  \
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,  \
      allocate_memory);

#define CREATE_FLOAT_BUFFER(sizes, allocate_memory) \
  vTensor(                                          \
      api::context(),                               \
      sizes,                                        \
      api::kFloat,                                  \
      api::StorageType::BUFFER,                     \
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, \
      allocate_memory);

#define DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(tensor) \
  api::StorageBuffer staging_buffer_##tensor(               \
      api::context(), api::kFloat, tensor.gpu_numel());     \
  record_nchw_to_image_op(                                  \
      api::context(), staging_buffer_##tensor.buffer(), tensor);

#define DEFINE_STAGING_BUFFER_AND_RECORD_FROM_GPU_FOR(tensor) \
  api::StorageBuffer staging_buffer_##tensor(                 \
      api::context(), api::kFloat, tensor.gpu_numel());       \
  record_image_to_nchw_op(                                    \
      api::context(), tensor, staging_buffer_##tensor.buffer());

//
// Operator Recording
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst);

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer);

void record_nchw_to_image_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst);

void record_image_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer);

void record_binary_op(
    api::Context* const context,
    const std::string& op_name,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst);

void execute_and_check_add(
    vTensor& a,
    vTensor& b,
    vTensor& c,
    float a_val,
    float b_val);

//
// Input & Output Utilities
//

inline void
fill_staging(api::StorageBuffer& staging, float val, int numel = -1) {
  if (numel < 0) {
    numel = staging.numel();
  }
  std::vector<float> data(numel);
  std::fill(data.begin(), data.end(), val);
  copy_ptr_to_staging(data.data(), staging, sizeof(float) * numel);
}

void fill_vtensor(vTensor& vten, std::vector<float>& data);

inline void fill_vtensor(vTensor& vten, float val) {
  std::vector<float> vten_data(vten.gpu_numel());
  std::fill(vten_data.begin(), vten_data.end(), val);

  fill_vtensor(vten, vten_data);
}

void fill_vtensor(ComputeGraph& graph, const IOValueRef idx, float val);

void extract_vtensor(vTensor& vten, std::vector<float>& data);

inline std::vector<float> extract_vtensor(vTensor& vten) {
  std::vector<float> data_out(vten.gpu_numel());
  extract_vtensor(vten, data_out);
  return data_out;
}

inline void
check_staging_buffer(api::StorageBuffer& staging, float val, int numel = -1) {
  if (numel < 0) {
    numel = staging.numel();
  }
  std::vector<float> data(numel);
  copy_staging_to_ptr(staging, data.data(), sizeof(float) * numel);

  for (const auto& d : data) {
    EXPECT_TRUE(d == val);
  }
}

//
// Context Management
//

void submit_to_gpu();

api::MemoryAllocation allocate_memory_for(const vTensor& vten);

VmaTotalStatistics get_vma_stats();

size_t get_vma_allocation_count();
