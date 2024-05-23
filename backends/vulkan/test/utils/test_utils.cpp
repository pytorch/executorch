/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/test/utils/test_utils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <cassert>

//
// Operator Recording Functions
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {SV(v_dst.packed_dim_whcn_idx())};

  context->submit_compute_job(
      get_nchw_to_tensor_shader(v_dst),
      pipeline_barrier,
      {uint32_t(v_dst.texel_numel()), 1, 1},
      {64, 1, 1},
      specialization_constants,
      VK_NULL_HANDLE,
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.sizes_ubo(),
      v_dst.texel_strides_ubo(),
      v_dst.ntexels_ubo());
}

void record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {SV(v_src.packed_dim_whcn_idx())};

  context->submit_compute_job(
      get_tensor_to_nchw_shader(v_src),
      pipeline_barrier,
      {uint32_t(v_src.texel_numel()), 1, 1},
      {64, 1, 1},
      specialization_constants,
      VK_NULL_HANDLE,
      v_src.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      dst_buffer,
      v_src.sizes_ubo(),
      v_src.texel_strides_ubo(),
      v_src.ntexels_ubo());
}

void record_nchw_to_image_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {SV(v_dst.packed_dim_whcn_idx())};

  context->submit_compute_job(
      get_nchw_to_tensor_shader(v_dst),
      pipeline_barrier,
      v_dst.image_extents(),
      adaptive_work_group_size(v_dst.image_extents()),
      specialization_constants,
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.sizes_ubo());
}

void record_image_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {SV(v_src.packed_dim_whcn_idx())};

  context->submit_compute_job(
      get_tensor_to_nchw_shader(v_src),
      pipeline_barrier,
      v_src.image_extents(),
      adaptive_work_group_size(v_src.image_extents()),
      specialization_constants,
      VK_NULL_HANDLE,
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      dst_buffer,
      v_src.sizes_ubo());
}

void record_conv2d_prepack_weights_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    const std::vector<int64_t>& original_sizes,
    const bool transposed) {
  api::PipelineBarrier pipeline_barrier{};

  std::string kernel_name;
  if (transposed) {
    kernel_name = "conv_transpose2d";
  } else {
    kernel_name = "conv2d";
  }
  kernel_name += "_prepack_weights";
  add_dtype_suffix(kernel_name, v_dst);
  api::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  api::UniformParamsBuffer original_sizes_ubo(
      context, api::utils::make_ivec4(original_sizes, /*reverse = */ true));

  api::SpecVarList specialization_constants = {};
  context->submit_compute_job(
      shader,
      pipeline_barrier,
      v_dst.image_extents(),
      adaptive_work_group_size(v_dst.image_extents()),
      specialization_constants,
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.sizes_ubo(),
      original_sizes_ubo.buffer());
}

void record_binary_op(
    api::Context* const context,
    const std::string& op_name,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst) {
  std::string kernel_name = "binary_" + op_name + "_nobroadcast__test";
  add_dtype_suffix(kernel_name, v_dst);

  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      v_dst.image_extents(),
      adaptive_work_group_size(v_dst.image_extents()),
      specialization_constants,
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_dst.sizes_ubo());
}

void execute_and_check_add(
    vTensor& a,
    vTensor& b,
    vTensor& c,
    float a_val,
    float b_val) {
  // Fill input tensors
  fill_vtensor(a, a_val);
  fill_vtensor(b, b_val);

  // a + b = c
  record_binary_op(api::context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Check output
  for (size_t i = 0; i < data_out.size(); ++i) {
    CHECK_VALUE(data_out, i, (a_val + b_val));
  }
}

void record_index_fill_buffer(api::Context* context, vTensor& v_ten) {
  std::string kernel_name("idx_fill_buffer");
  switch (v_ten.dtype()) {
    case api::kFloat:
      kernel_name += "_float";
      break;
    case api::kHalf:
      kernel_name += "_half";
      break;
    case api::kQInt8:
      kernel_name += "_int8";
      break;
    case api::kQUInt8:
      kernel_name += "_uint8";
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
      break;
  }

  api::UniformParamsBuffer params(api::context(), int32_t(v_ten.numel()));

  {
    api::PipelineBarrier pipeline_barrier{};
    api::SpecVarList specialization_constants = {};
    api::context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {uint32_t(v_ten.texel_numel()), 1, 1},
        {64, 1, 1},
        specialization_constants,
        VK_NULL_HANDLE,
        v_ten.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
  }
}

void record_scalar_add_buffer(
    api::Context* context,
    vTensor& v_ten,
    float offset) {
  api::PipelineBarrier pipeline_barrier{};
  api::SpecVarList specialization_constants = {SV(offset)};
  api::context()->submit_compute_job(
      VK_KERNEL(scalar_add_buffer),
      pipeline_barrier,
      {uint32_t(v_ten.texel_numel()), 1, 1},
      {64, 1, 1},
      specialization_constants,
      VK_NULL_HANDLE,
      v_ten.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_ten.ntexels_ubo());
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

void fill_vtensor(vTensor& vten, float val, bool iota) {
  std::vector<float> vten_data(vten.gpu_numel());
  if (iota) {
    std::iota(vten_data.begin(), vten_data.end(), val);
  } else {
    std::fill(vten_data.begin(), vten_data.end(), val);
  }

  fill_vtensor(vten, vten_data);
}

void fill_vtensor(
    ComputeGraph& graph,
    const IOValueRef idx,
    float val,
    bool iota) {
  std::vector<float> data(graph.get_tensor(idx.value)->gpu_numel());
  if (iota) {
    std::iota(data.begin(), data.end(), val);
  } else {
    std::fill(data.begin(), data.end(), val);
  }

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

api::Allocation allocate_memory_for(const vTensor& vten) {
  return api::context()->adapter_ptr()->vma().create_allocation(
      vten.get_memory_requirements(), vten.get_allocation_create_info());
}

VmaTotalStatistics get_vma_stats() {
  return api::context()->adapter_ptr()->vma().get_memory_statistics();
}

size_t get_vma_allocation_count() {
  return get_vma_stats().total.statistics.allocationCount;
}

//
// Graph Test Utilities
//

void execute_graph_and_check_output(
    ComputeGraph& graph,
    std::vector<float> input_vals,
    std::vector<float> expected_outputs) {
  assert(input_vals.size() == graph.inputs().size());
  assert(expected_outputs.size() == graph.outputs().size());

  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fill_vtensor(graph, graph.inputs().at(i), input_vals.at(i));
  }

  graph.execute();

  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    IOValueRef out_ioval = graph.outputs().at(i);
    vTensorPtr t_out = graph.get_tensor(out_ioval.value);

    std::vector<float> output_data(t_out->gpu_numel());
    graph.copy_from_staging(
        out_ioval.staging, output_data.data(), output_data.size());

    for (size_t j = 0; j < t_out->numel(); ++j) {
      CHECK_VALUE(output_data, j, expected_outputs.at(i));
    }
  }
}
