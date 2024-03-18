/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/test/utils/test_utils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

//
// Operator Recording Functions
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_dst.gpu_numel());
  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_dst.get_cpu_buffer_metadata());
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_dst.buffer_metadata(),
      src_buffer,
      cpu_buffer_metadata.buffer());
}

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_src.numel());
  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());
  api::PipelineBarrier pipeline_barrier{};

  return context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

void record_nchw_to_image_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  api::PipelineBarrier pipeline_barrier{};
  api::ShaderInfo compute_shader =
      VK_KERNEL(nchw_to_image3d__test_C_packed_half);
  if (v_dst.image().format() == VK_FORMAT_R32G32B32A32_SFLOAT) {
    compute_shader = VK_KERNEL(nchw_to_image3d__test_C_packed_float);
  }
  context->submit_compute_job(
      compute_shader,
      pipeline_barrier,
      v_dst.virtual_extents(),
      adaptive_work_group_size(v_dst.virtual_extents()),
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.gpu_sizes_ubo()->buffer(),
      v_dst.cpu_sizes_ubo()->buffer());
}

void record_image_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  api::ShaderInfo compute_shader =
      VK_KERNEL(image3d_to_nchw__test_C_packed_half);
  if (v_src.image().format() == VK_FORMAT_R32G32B32A32_SFLOAT) {
    compute_shader = VK_KERNEL(image3d_to_nchw__test_C_packed_float);
  }
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      compute_shader,
      pipeline_barrier,
      v_src.virtual_extents(),
      adaptive_work_group_size(v_src.virtual_extents()),
      VK_NULL_HANDLE,
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      dst_buffer,
      v_src.gpu_sizes_ubo()->buffer(),
      v_src.cpu_sizes_ubo()->buffer());
}

void record_binary_op(
    api::Context* const context,
    const std::string& op_name,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst) {
  std::stringstream kernel_name;
  kernel_name << "binary_" << op_name << "_nobroadcast__test";
  apply_dtype_suffix(kernel_name, v_dst);

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name.str()),
      pipeline_barrier,
      v_dst.virtual_extents(),
      adaptive_work_group_size(v_dst.virtual_extents()),
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_dst.extents_ubo()->buffer());
}

void execute_and_check_add(
    vTensor& a,
    vTensor& b,
    vTensor& c,
    float a_val,
    float b_val) {
  // Add shader kernel
  api::ShaderInfo kernel = VK_KERNEL(binary_add_nobroadcast__test_half);
  if (c.image().format() == VK_FORMAT_R32G32B32A32_SFLOAT) {
    kernel = VK_KERNEL(nchw_to_image3d__test_C_packed_float);
  }

  // Fill input tensors
  fill_vtensor(a, a_val);
  fill_vtensor(b, b_val);

  // a + b = c
  record_binary_op(api::context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Check output
  for (const auto& d : data_out) {
    EXPECT_TRUE(d == (a_val + b_val));
  }
}

//
// Input & Output Utilities
//

void fill_vtensor(vTensor& vten, std::vector<float>& data) {
  api::StorageBuffer staging_buffer(api::context(), api::kFloat, data.size());

  copy_ptr_to_staging(data.data(), staging_buffer, vten.gpu_nbytes());

  if (vten.storage_type() == api::StorageType::BUFFER) {
    record_nchw_to_buffer_op(api::context(), staging_buffer.buffer(), vten);
  } else {
    record_nchw_to_image_op(api::context(), staging_buffer.buffer(), vten);
  }
}

void fill_vtensor(ComputeGraph& graph, const IOValueRef idx, float val) {
  std::vector<float> data(graph.get_val(idx.value).toTensor().gpu_numel());
  std::fill(data.begin(), data.end(), val);

  graph.copy_into_staging(idx.staging, data.data(), data.size());
}

void extract_vtensor(vTensor& vten, std::vector<float>& data) {
  api::StorageBuffer staging_buffer(
      api::context(), api::kFloat, vten.gpu_numel());

  if (vten.storage_type() == api::StorageType::BUFFER) {
    record_buffer_to_nchw_op(api::context(), vten, staging_buffer.buffer());
  } else {
    record_image_to_nchw_op(api::context(), vten, staging_buffer.buffer());
  }

  api::VulkanFence fence = api::context()->fences().get_fence();
  api::context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();

  copy_staging_to_ptr(staging_buffer, data.data(), vten.gpu_nbytes());
}

//
// Context Management
//

void submit_to_gpu() {
  api::VulkanFence fence = api::context()->fences().get_fence();
  api::context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
}

api::MemoryAllocation allocate_memory_for(const vTensor& vten) {
  return api::context()->adapter_ptr()->vma().create_allocation(
      vten.get_memory_requirements(), vten.get_allocation_create_info());
}

VmaTotalStatistics get_vma_stats() {
  return api::context()->adapter_ptr()->vma().get_memory_statistics();
}

size_t get_vma_allocation_count() {
  return get_vma_stats().total.statistics.allocationCount;
}
