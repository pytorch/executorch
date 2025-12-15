/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/test/utils/test_utils.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <cassert>
#include <random>
#include <string>

using namespace vkcompute;

bool is_bitw8(vkapi::ScalarType dtype) {
  return dtype == vkapi::kByte || dtype == vkapi::kChar ||
      dtype == vkapi::kQInt8 || dtype == vkapi::kQUInt8;
}

vkapi::ShaderInfo get_nchw_to_tensor_shader(
    const api::vTensor& v_dst,
    bool int8_buffer_enabled,
    bool push_constant_variant) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  if (is_bitw8(v_dst.dtype()) && v_dst.storage_type() != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "nchw_to_bitw8_image_nobitw8buffer";
    if (!push_constant_variant) {
      kernel_name += "_no_pc";
    }
    add_storage_type_suffix(kernel_name, v_dst.storage_type());
    add_dtype_suffix(kernel_name, v_dst.dtype());
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (v_dst.storage_type() == utils::kBuffer) {
    kernel_name = "nchw_to_buffer";
    add_dtype_suffix(kernel_name, v_dst.dtype());
    add_dtype_suffix(kernel_name, v_dst.dtype());
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "nchw_to_image";
  if (!push_constant_variant) {
    kernel_name += "_no_pc";
  }
  add_storage_type_suffix(kernel_name, v_dst.storage_type());
  add_dtype_suffix(kernel_name, v_dst.dtype());
  add_dtype_suffix(kernel_name, v_dst.dtype());

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo get_tensor_to_nchw_shader(
    const api::vTensor& v_src,
    bool int8_buffer_enabled,
    bool push_constant_variant) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  if (is_bitw8(v_src.dtype()) && v_src.storage_type() != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "bitw8_image_to_nchw_nobitw8buffer";
    if (!push_constant_variant) {
      kernel_name += "_no_pc";
    }
    add_storage_type_suffix(kernel_name, v_src.storage_type());
    add_dtype_suffix(kernel_name, v_src.dtype());
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (v_src.storage_type() == utils::kBuffer) {
    kernel_name = "buffer_to_nchw";
    add_dtype_suffix(kernel_name, v_src.dtype());
    add_dtype_suffix(kernel_name, v_src.dtype());
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "image_to_nchw";
  if (!push_constant_variant) {
    kernel_name += "_no_pc";
  }
  add_storage_type_suffix(kernel_name, v_src.storage_type());
  add_dtype_suffix(kernel_name, v_src.dtype());
  add_dtype_suffix(kernel_name, v_src.dtype());

  return VK_KERNEL_FROM_STR(kernel_name);
}
//
// Operator Recording Functions
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    vkapi::VulkanBuffer& src_buffer,
    api::vTensor& v_dst) {
  vkapi::PipelineBarrier pipeline_barrier{};
  vkapi::SpecVarList specialization_constants = {v_dst.hashed_layout()};

  context->submit_compute_job(
      get_nchw_to_tensor_shader(v_dst, true, false),
      pipeline_barrier,
      {uint32_t(v_dst.numel()), 1, 1},
      {64, 1, 1},
      specialization_constants,
      VK_NULL_HANDLE,
      0,
      v_dst.buffer(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.buffer_meta_ubo());
}

void record_buffer_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    vkapi::VulkanBuffer& dst_buffer) {
  vkapi::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      get_tensor_to_nchw_shader(v_src, true, false),
      pipeline_barrier,
      {uint32_t(v_src.numel()), 1, 1},
      {64, 1, 1},
      {},
      VK_NULL_HANDLE,
      0,
      dst_buffer,
      v_src.buffer(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      v_src.buffer_meta_ubo());
}

void record_nchw_to_image_op(
    api::Context* const context,
    vkapi::VulkanBuffer& src_buffer,
    api::vTensor& v_dst) {
  vkapi::PipelineBarrier pipeline_barrier{};
  vkapi::SpecVarList specialization_constants = {v_dst.hashed_layout()};

  context->submit_compute_job(
      get_nchw_to_tensor_shader(
          v_dst,
          context->adapter_ptr()->has_full_int8_buffers_support(),
          false),
      pipeline_barrier,
      v_dst.logical_limits(),
      adaptive_work_group_size(v_dst.logical_limits()),
      specialization_constants,
      VK_NULL_HANDLE,
      0,
      v_dst.image(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE),
      src_buffer,
      v_dst.sizes_ubo());
}

void record_image_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    vkapi::VulkanBuffer& dst_buffer) {
  vkapi::PipelineBarrier pipeline_barrier{};
  vkapi::SpecVarList specialization_constants = {v_src.hashed_layout()};

  context->submit_compute_job(
      get_tensor_to_nchw_shader(v_src, true, false),
      pipeline_barrier,
      v_src.logical_limits(),
      adaptive_work_group_size(v_src.logical_limits()),
      specialization_constants,
      VK_NULL_HANDLE,
      0,
      dst_buffer,
      v_src.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      v_src.sizes_ubo());
}

void record_bitw8_image_to_nchw_nobitw8buffer_op(
    api::Context* const context,
    api::vTensor& v_src,
    api::StagingBuffer& dst_buffer) {
  vkapi::PipelineBarrier pipeline_barrier{};
  uint32_t buffer_len = utils::safe_downcast<uint32_t>(dst_buffer.numel() / 4);
  utils::uvec3 global_wg_size = {buffer_len, 1, 1};

  std::string kernel_name = "bitw8_image_to_nchw_nobitw8buffer_no_pc";
  add_storage_type_suffix(kernel_name, v_src.storage_type());
  add_dtype_suffix(kernel_name, v_src.dtype());

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      global_wg_size,
      adaptive_work_group_size(global_wg_size),
      {v_src.hashed_layout()},
      VK_NULL_HANDLE,
      0,
      dst_buffer.buffer(),
      v_src.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      v_src.sizes_ubo(),
      v_src.numel_ubo());
}

void record_binary_op(
    api::Context* const context,
    const std::string& op_name,
    api::vTensor& v_in1,
    api::vTensor& v_in2,
    api::vTensor& v_dst) {
  std::string kernel_name = "binary_" + op_name + "_nobroadcast__test";
  add_dtype_suffix(kernel_name, v_dst.dtype());

  vkapi::PipelineBarrier pipeline_barrier{};
  vkapi::SpecVarList specialization_constants = {};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      v_dst.logical_limits(),
      adaptive_work_group_size(v_dst.logical_limits()),
      specialization_constants,
      VK_NULL_HANDLE,
      0,
      v_dst.image(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      v_dst.sizes_ubo());
}

void execute_and_check_add(
    api::vTensor& a,
    api::vTensor& b,
    api::vTensor& c,
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

void record_index_fill_buffer(api::Context* context, api::vTensor& v_ten) {
  std::string kernel_name("idx_fill_buffer");
  switch (v_ten.dtype()) {
    case vkapi::kFloat:
      kernel_name += "_float";
      break;
    case vkapi::kHalf:
      kernel_name += "_half";
      break;
    case vkapi::kQInt8:
      kernel_name += "_int8";
      break;
    case vkapi::kQUInt8:
      kernel_name += "_uint8";
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
      break;
  }

  api::ParamsBuffer params(api::context(), int32_t(v_ten.numel()));

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {};
    api::context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {uint32_t(v_ten.numel()), 1, 1},
        {64, 1, 1},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        v_ten.buffer(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::READ),
        params.buffer());
  }
}

void record_scalar_add_buffer(
    api::Context* context,
    api::vTensor& v_ten,
    float offset) {
  vkapi::PipelineBarrier pipeline_barrier{};
  vkapi::SpecVarList specialization_constants = {SV(offset)};
  std::string kernel = "scalar_add_buffer";
  add_dtype_suffix(kernel, v_ten.dtype());
  api::context()->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel),
      pipeline_barrier,
      {uint32_t(v_ten.numel()), 1, 1},
      {64, 1, 1},
      specialization_constants,
      VK_NULL_HANDLE,
      0,
      v_ten.buffer(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::READ | vkapi::MemoryAccessType::WRITE),
      v_ten.numel_ubo());
}

void record_reference_matmul(
    api::Context* context,
    api::vTensor& out,
    api::vTensor& mat1,
    api::vTensor& mat2) {
  vkapi::PipelineBarrier pipeline_barrier{};
  api::context()->submit_compute_job(
      VK_KERNEL(reference_matmul),
      pipeline_barrier,
      {uint32_t(out.size(1)), uint32_t(out.size(0)), 1},
      {64, 1, 1},
      {},
      VK_NULL_HANDLE,
      0,
      out.buffer(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE),
      mat1.buffer(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      mat2.buffer(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      out.sizes_ubo(),
      out.strides_ubo(),
      mat1.sizes_ubo(),
      mat1.strides_ubo(),
      mat2.sizes_ubo(),
      mat2.strides_ubo());
}

void record_matmul_texture3d(
    api::Context* context,
    api::vTensor& out,
    api::vTensor& mat1,
    api::vTensor& mat2) {
  std::string kernel_name = "matmul_naive";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, out.storage_type());
  add_dtype_suffix(kernel_name, out.dtype());

  utils::uvec3 global_wg_size = out.logical_limits();

  vkapi::PipelineBarrier pipeline_barrier{};
  api::context()->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      global_wg_size,
      {8, 8, 1},
      {out.hashed_layout(), mat1.hashed_layout(), mat2.hashed_layout()},
      VK_NULL_HANDLE,
      0,
      out.image(
          pipeline_barrier,
          vkapi::PipelineStage::COMPUTE,
          vkapi::MemoryAccessType::WRITE),
      mat1.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      mat2.image(pipeline_barrier, vkapi::PipelineStage::COMPUTE),
      out.sizes_ubo(),
      out.logical_limits_ubo(),
      mat1.sizes_ubo(),
      mat2.sizes_ubo());
}

//
// Input & Output Utilities
//

#define FORALL_SUPPORTED_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int32_t, Int)                 \
  _(executorch::aten::Half, Half) \
  _(float, Float)                 \
  _(int8_t, QInt8)

void fill_vtensor(api::vTensor& vten, std::vector<float>& data) {
  api::StagingBuffer staging_buffer(api::context(), vten.dtype(), data.size(), vkapi::CopyDirection::HOST_TO_DEVICE);

#define CASE(ctype, name)                                     \
  case vkapi::ScalarType::name: {                             \
    std::vector<ctype> data_converted;                        \
    data_converted.resize(data.size());                       \
    for (int i = 0; i < data.size(); ++i) {                   \
      data_converted[i] = ctype(data[i]);                     \
    }                                                         \
    staging_buffer.copy_from(                                 \
        data_converted.data(), vten.staging_buffer_nbytes()); \
  } break;

  switch (vten.dtype()) {
    FORALL_SUPPORTED_TYPES(CASE)
    default:
      VK_THROW("Unsupported dtype");
  }

#undef CASE

  if (vten.storage_type() == utils::StorageType::BUFFER) {
    record_nchw_to_buffer_op(api::context(), staging_buffer.buffer(), vten);
  } else {
    record_nchw_to_image_op(api::context(), staging_buffer.buffer(), vten);
  }
}

void fill_vtensor(api::vTensor& vten, float val, bool iota) {
  std::vector<float> vten_data(vten.staging_buffer_numel());
  if (iota) {
    std::iota(vten_data.begin(), vten_data.end(), val);
  } else {
    std::fill(vten_data.begin(), vten_data.end(), val);
  }

  fill_vtensor(vten, vten_data);
}

std::vector<float> create_random_float_buffer(
    const size_t numel,
    const float min,
    const float max) {
  std::vector<float> data(numel);
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(min, max);

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = dist(rng);
  }
  return data;
}

std::vector<uint8_t> create_random_uint8_buffer(
    const size_t numel,
    const uint8_t min,
    const uint8_t max) {
  std::vector<uint8_t> data(numel);
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(min, max);

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = (uint8_t)dist(rng);
  }
  return data;
}

void fill_vtensor(
    ComputeGraph& graph,
    const IOValueRef idx,
    float val,
    bool iota) {
  std::vector<float> data(graph.numel_of(idx.value));
  if (graph.storage_type_of(idx.value) != utils::kBuffer) {
    data.resize(graph.staging_buffer_numel_of(idx.value));
  }
  if (iota) {
    std::iota(data.begin(), data.end(), val);
  } else {
    std::fill(data.begin(), data.end(), val);
  }

  graph.copy_into_staging(idx.staging, data.data(), data.size());
}

void extract_vtensor(api::vTensor& vten, std::vector<float>& data) {
  api::StagingBuffer staging_buffer(
      api::context(), vten.dtype(), vten.staging_buffer_numel(), vkapi::CopyDirection::DEVICE_TO_HOST);

  if (vten.storage_type() == utils::StorageType::BUFFER) {
    record_buffer_to_nchw_op(api::context(), vten, staging_buffer.buffer());
  } else {
    record_image_to_nchw_op(api::context(), vten, staging_buffer.buffer());
  }

  vkapi::VulkanFence fence = api::context()->fences().get_fence();
  api::context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();

#define CASE(ctype, name)                                     \
  case vkapi::ScalarType::name: {                             \
    std::vector<ctype> data_converted(data.size());           \
    staging_buffer.copy_to(                                   \
        data_converted.data(), vten.staging_buffer_nbytes()); \
    for (int i = 0; i < data.size(); ++i) {                   \
      data[i] = float(data_converted[i]);                     \
    }                                                         \
  } break;

  switch (vten.dtype()) {
    FORALL_SUPPORTED_TYPES(CASE)
    default:
      VK_THROW("Unsupported dtype");
  }

#undef CASE
}

//
// Context Management
//

void submit_to_gpu() {
  vkapi::VulkanFence fence = api::context()->fences().get_fence();
  api::context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
}

vkapi::Allocation allocate_memory_for(const api::vTensor& vten) {
  VmaAllocationCreateInfo alloc_create_info =
      api::context()->adapter_ptr()->vma().gpuonly_resource_create_info();
  return api::context()->adapter_ptr()->vma().create_allocation(
      vten.get_memory_requirements(), alloc_create_info);
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
    std::vector<float> output_data(
        graph.staging_buffer_numel_of(out_ioval.value));
    graph.copy_from_staging(
        out_ioval.staging, output_data.data(), output_data.size());

    for (size_t j = 0; j < graph.numel_of(out_ioval.value); ++j) {
      CHECK_VALUE(output_data, j, expected_outputs.at(i));
    }
  }
}

vkcompute::ComputeGraph build_mm_graph(
    int B,
    int M,
    int K,
    int N,
    vkcompute::vkapi::ScalarType dtype,
    vkcompute::utils::StorageType in_out_stype,
    vkcompute::utils::GPUMemoryLayout memory_layout,
    const std::vector<float>& mat2_data,
    const bool prepack_mat2) {
  using namespace vkcompute;
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  ComputeGraph graph(config);

  std::vector<int64_t> mat1_size = {M, K};
  std::vector<int64_t> mat2_size = {K, N};
  std::vector<int64_t> out_size = {M, N};
  if (B > 1) {
    mat1_size.resize(3);
    mat1_size = {B, M, K};
    mat2_size.resize(3);
    mat2_size = {B, K, N};
    out_size.resize(3);
    out_size = {B, M, N};
  }

  IOValueRef mat1 =
      graph.add_input_tensor(mat1_size, dtype, in_out_stype, memory_layout);
  IOValueRef mat2{};

  ValueRef mat2_w = graph.add_tensorref(mat2_size, dtype, mat2_data.data());

  if (prepack_mat2) {
    mat2.value = mat2_w;
  } else {
    mat2.value =
        graph.add_tensor(mat2_size, dtype, in_out_stype, memory_layout);
    mat2.staging = graph.set_input_tensor(mat2.value);
  }

  IOValueRef out;
  out.value = graph.add_tensor(out_size, dtype, in_out_stype, memory_layout);

  VK_GET_OP_FN("aten.mm.default")(graph, {mat1.value, mat2.value, out.value});

  out.staging = graph.set_output_tensor(out.value);

  return graph;
}

bool check_close(float a, float b, float atol, float rtol) {
  float max = std::max(std::abs(a), std::abs(b));
  float diff = std::abs(a - b);
  return diff <= (atol + rtol * max);
}
