/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/TensorMeta.h>
#include <executorch/backends/webgpu/runtime/ops/slice/slice_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct SliceParams {
  uint32_t dim;
  uint32_t start;
  uint32_t step;
  uint32_t _pad;
};

// aten.split_with_sizes_copy: N contiguous chunks via per-output slice gather.
void split_with_sizes_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 4) {
    throw std::runtime_error("WebGPU split_with_sizes: expected >=4 args");
  }
  const int in_id = args.at(0);
  const std::vector<int64_t>& sizes = graph.get_int_list(args.at(1));
  int64_t dim = graph.get_int(args.at(2));
  const std::vector<int>& outs = graph.get_value_list(args.at(args.size() - 1));

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const int in_ndim = static_cast<int>(in_tensor.dims.size());
  if (dim < 0) {
    dim += in_ndim;
  }
  if (dim < 0 || dim >= in_ndim) {
    throw std::runtime_error("split_with_sizes: dim out of range");
  }
  if (outs.size() != sizes.size()) {
    throw std::runtime_error("split_with_sizes: outputs != sizes count");
  }

  // Validate the split contract up front (before any dispatch): each size is
  // non-negative and they sum to the split dim's extent -- a negative or
  // oversize value would otherwise wrap when cast to u32 into an out-of-bounds
  // gather offset.
  int64_t sizes_sum = 0;
  for (int64_t s : sizes) {
    if (s < 0) {
      throw std::runtime_error("split_with_sizes: negative split size");
    }
    sizes_sum += s;
  }
  if (sizes_sum != in_tensor.dims[dim]) {
    throw std::runtime_error(
        "split_with_sizes: sizes must sum to the split dim extent");
  }

  TensorMeta in_meta;
  fill_tensor_meta(in_tensor, &in_meta);
  if (in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("split_with_sizes: non-fp32 input");
  }

  uint32_t start = 0;
  for (size_t i = 0; i < outs.size(); i++) {
    const auto& out_tensor = graph.get_tensor(outs[i]);
    TensorMeta out_meta;
    fill_tensor_meta(out_tensor, &out_meta);
    if (out_tensor.nbytes !=
        static_cast<size_t>(out_meta.numel) * sizeof(float)) {
      throw std::runtime_error("split_with_sizes: non-fp32 output");
    }

    SliceParams params = {};
    params.dim = static_cast<uint32_t>(dim);
    params.start = start;
    params.step = 1u;
    start += static_cast<uint32_t>(sizes[i]);

    uint32_t wg_size =
        utils::clamp_workgroup_size(device, kSliceWorkgroupSizeX);
    uint32_t workgroup_count = utils::compute_1d_workgroup_count(
        device, out_meta.numel, wg_size, "split_with_sizes");

    WGPUConstantEntry wg_size_constant = {};
    wg_size_constant.key = {"wg_size", WGPU_STRLEN};
    wg_size_constant.value = static_cast<double>(wg_size);

    WGPUBuffer out_meta_buf =
        utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
    WGPUBuffer in_meta_buf =
        utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
    WGPUBuffer params_buf =
        utils::make_uniform(device, &params, sizeof(SliceParams));
    graph.add_uniform_buffer_bytes(
        2 * sizeof(TensorMeta) + sizeof(SliceParams));

    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kSliceWGSL,
        {
            {0,
             WGPUBufferBindingType_ReadOnlyStorage,
             in_tensor.buffer,
             in_tensor.nbytes},
            {1,
             WGPUBufferBindingType_Storage,
             out_tensor.buffer,
             out_tensor.nbytes},
            {2,
             WGPUBufferBindingType_Uniform,
             out_meta_buf,
             sizeof(TensorMeta)},
            {3, WGPUBufferBindingType_Uniform, in_meta_buf, sizeof(TensorMeta)},
            {4, WGPUBufferBindingType_Uniform, params_buf, sizeof(SliceParams)},
        },
        &wg_size_constant,
        1);

    graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

    graph.own_uniform_buffer(out_meta_buf);
    graph.own_uniform_buffer(in_meta_buf);
    graph.own_uniform_buffer(params_buf);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.split_with_sizes_copy.default, split_with_sizes_impl);
}

} // namespace executorch::backends::webgpu
