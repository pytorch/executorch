/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

using namespace vkcompute;

#define CREATE_FLOAT_TEXTURE(sizes, allocate_memory)  \
  api::vTensor(                                       \
      api::context(),                                 \
      sizes,                                          \
      vkapi::kFloat,                                  \
      utils::StorageType::TEXTURE_3D,                 \
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, \
      allocate_memory);

#define CREATE_FLOAT_BUFFER(sizes, allocate_memory) \
  api::vTensor(                                     \
      api::context(),                               \
      sizes,                                        \
      vkapi::kFloat,                                \
      utils::StorageType::BUFFER,                   \
      utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED,  \
      allocate_memory);

#define DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(tensor)          \
  api::StagingBuffer staging_buffer_##tensor(                        \
      api::context(), vkapi::kFloat, tensor.staging_buffer_numel()); \
  record_nchw_to_image_op(                                           \
      api::context(), staging_buffer_##tensor.buffer(), tensor);

#define DEFINE_STAGING_BUFFER_AND_RECORD_FROM_GPU_FOR(tensor)        \
  api::StagingBuffer staging_buffer_##tensor(                        \
      api::context(), vkapi::kFloat, tensor.staging_buffer_numel()); \
  record_image_to_nchw_op(                                           \
      api::context(), tensor, staging_buffer_##tensor.buffer());

#define CHECK_VALUE(data, idx, expected)                          \
  do {                                                            \
    if (data[idx] != expected) {                                  \
      std::cout << "Output at [" << idx << "] = " << data[idx]    \
                << ", does not match expected value " << expected \
                << std::endl;                                     \
    }                                                             \
    ASSERT_TRUE(data[idx] == expected);                           \
  } while (false)

//
// Operator Recording
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    vkapi::VulkanBuffer& src_buffer,
    api::vTensor& v_dst);

void record_buffer_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    vkapi::VulkanBuffer& dst_buffer);

void record_nchw_to_image_op(
    api::Context* const context,
    vkapi::VulkanBuffer& src_buffer,
    api::vTensor& v_dst);

void record_image_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    vkapi::VulkanBuffer& dst_buffer);

void record_int8_image_to_nchw_noint8_op(
    api::Context* const context,
    api::vTensor& v_src,
    api::StagingBuffer& dst_buffer);

void record_conv2d_prepack_weights_op(
    api::Context* const context,
    vkapi::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    const std::vector<int64_t>& original_sizes,
    const bool transposed);

void record_binary_op(
    api::Context* const context,
    const std::string& op_name,
    api::vTensor& v_in1,
    api::vTensor& v_in2,
    api::vTensor& v_dst);

void execute_and_check_add(
    api::vTensor& a,
    api::vTensor& b,
    api::vTensor& c,
    float a_val,
    float b_val);

void record_index_fill_buffer(api::Context* const context, api::vTensor& v_ten);

void record_scalar_add_buffer(
    api::Context* context,
    api::vTensor& v_ten,
    float offset);

void record_reference_matmul(
    api::Context* context,
    api::vTensor& out,
    api::vTensor& mat1,
    api::vTensor& mat2);

//
// Input & Output Utilities
//

inline void
fill_staging(api::StagingBuffer& staging, float val, int numel = -1) {
  if (numel < 0) {
    numel = staging.numel();
  }
  std::vector<float> data(numel);
  std::fill(data.begin(), data.end(), val);
  staging.copy_from(data.data(), sizeof(float) * numel);
}

void fill_vtensor(api::vTensor& vten, std::vector<float>& data);

void fill_vtensor(api::vTensor& vten, float val, bool iota = false);

std::vector<float> create_random_float_buffer(
    const size_t numel,
    const float min = 0,
    const float max = 1);

std::vector<uint8_t> create_random_uint8_buffer(
    const size_t numel,
    const uint8_t min = 0,
    const uint8_t max = 255);

void fill_vtensor(
    ComputeGraph& graph,
    const IOValueRef idx,
    float val,
    bool iota = false);

void extract_vtensor(api::vTensor& vten, std::vector<float>& data);

inline std::vector<float> extract_vtensor(api::vTensor& vten) {
  std::vector<float> data_out(vten.staging_buffer_numel());
  extract_vtensor(vten, data_out);
  return data_out;
}

inline void
check_staging_buffer(api::StagingBuffer& staging, float val, int numel = -1) {
  if (numel < 0) {
    numel = staging.numel();
  }
  std::vector<float> data(numel);
  staging.copy_to(data.data(), sizeof(float) * numel);

  for (size_t i = 0; i < data.size(); ++i) {
    CHECK_VALUE(data, i, val);
  }
}

inline int64_t get_buf_idx(
    ComputeGraph& graph,
    IOValueRef ref,
    const std::vector<int64_t>& tensor_coor) {
  vTensorPtr vten_ptr = graph.get_tensor(ref.value);

  const std::vector<int64_t>& sizes = vten_ptr->sizes();

  int64_t c = dim_at<kChannel4D>(sizes);
  int64_t h = dim_at<kHeight4D>(sizes);
  int64_t w = dim_at<kWidth4D>(sizes);

  int64_t ni = dim_at<kBatch4D>(tensor_coor);
  int64_t ci = dim_at<kChannel4D>(tensor_coor);
  int64_t hi = dim_at<kHeight4D>(tensor_coor);
  int64_t wi = dim_at<kWidth4D>(tensor_coor);

  return (ni * c * h * w + ci * h * w + hi * w + wi);
}

//
// Context Management
//

void submit_to_gpu();

vkapi::Allocation allocate_memory_for(const api::vTensor& vten);

VmaTotalStatistics get_vma_stats();

size_t get_vma_allocation_count();

//
// Graph Test Utilities
//

void execute_graph_and_check_output(
    ComputeGraph& graph,
    std::vector<float> input_vals,
    std::vector<float> expected_outputs);

//
// Debugging Utilities
//

#define PRINT_DATA(vec)        \
  do {                         \
    std::cout << #vec << ": "; \
    print_vector(vec);         \
  } while (false);

#define PRINT_DATA_RANGE(vec, start, range)                                \
  do {                                                                     \
    std::cout << #vec << "[" << start << ", " << (start + range) << "]: "; \
    print_vector(vec, start, range);                                       \
  } while (false);

template <typename T>
void print_vector(
    const std::vector<T>& data,
    size_t start = 0,
    size_t range = 20) {
  size_t end = data.size();
  if (range >= 1) {
    end = std::min(data.size(), start + range);
  }
  for (size_t i = start; i < end; ++i) {
    std::cout << data.at(i) << ", ";
  }
  std::cout << std::endl;
}

//
// Misc. Utilities
//

bool check_close(float a, float b, float atol = 1e-4, float rtol = 1e-5);
