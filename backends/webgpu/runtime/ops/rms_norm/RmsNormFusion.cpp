/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/rms_norm/RmsNormFusion.h>

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_vec4_add_scale_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_vec4_add_wgsl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace executorch::backends::webgpu {
namespace fusion {

namespace {

using Record = WebGPUGraph::RmsFusionSite;

uint64_t numel_of_tensor(const WebGPUTensor& t) {
  return utils::numel_of(t.dims);
}

bool still_last(const WebGPUGraph& graph, const Record& r) {
  return r.valid && r.dispatch_index + 1u == graph.num_dispatches();
}

void resize_fused_add_outputs(
    WebGPUGraph& graph,
    int rms_in_id,
    int resid_id,
    int addout_id,
    int scale_id = -1,
    int scaleout_id = -1) {
  const auto& rms_dims = graph.cur_dims(rms_in_id);
  RmsResizeCheck check;
  check.residual_shape_matches = graph.cur_dims(resid_id) == rms_dims;
  check.has_scale = scale_id >= 0;
  check.scale_numel =
      check.has_scale ? utils::numel_of(graph.cur_dims(scale_id)) : 0u;
  if (const char* reason = rms_resize_reject_reason(check)) {
    throw std::runtime_error(std::string("WebGPU rms fusion: ") + reason);
  }
  graph.set_cur_dims(addout_id, rms_dims);
  if (scaleout_id >= 0) {
    graph.set_cur_dims(scaleout_id, rms_dims);
  }
}

// Replace dispatches_[idx]'s pipeline + bind group only after the complete
// replacement bundle has been created successfully.
void install(
    WebGPUGraph& graph,
    size_t dispatch_idx,
    const char* wgsl,
    const std::vector<utils::BindingSpec>& bindings) {
  utils::ComputePipelineBundle replacement =
      utils::make_compute_pipeline(graph.device(), wgsl, bindings);

  WebGPUDispatch& d = graph.dispatch_at(dispatch_idx);
  WGPUComputePipeline old_pipeline = d.pipeline;
  WGPUBindGroup old_bind_group = d.bind_group;
  d.pipeline = replacement.pipeline;
  d.bind_group = replacement.bind_group;
  if (old_pipeline != nullptr) {
    wgpuComputePipelineRelease(old_pipeline);
  }
  if (old_bind_group != nullptr) {
    wgpuBindGroupRelease(old_bind_group);
  }
}

} // namespace

void invalidate_record(WebGPUGraph& graph) {
  graph.clear_rms_fusion_site();
}

void record_rms_norm(
    WebGPUGraph& graph,
    int in_id,
    int weight_id,
    int out_id,
    uint32_t num_rows,
    uint32_t row_width,
    size_t dispatch_idx,
    WGPUBuffer params_buf,
    WGPUBindGroup /*bind_group*/) {
  Record r = {};
  r.in_id = in_id;
  r.weight_id = weight_id;
  r.out_id = out_id;
  r.num_rows = num_rows;
  r.row_width = row_width;
  r.dispatch_index = dispatch_idx;
  r.params_buffer = params_buf;
  graph.offer_rms_fusion_site(std::move(r));
}

bool try_fuse_add(
    WebGPUGraph& graph,
    int in1_id,
    int in2_id,
    float alpha,
    int out_id) {
  Record r = graph.rms_fusion_site();
  if (!still_last(graph, r) || r.add_fused) {
    return false;
  }

  const bool first_is_rms = (in1_id == r.out_id);
  const bool second_is_rms = (in2_id == r.out_id);
  const int resid_id = first_is_rms ? in2_id : in1_id;

  const WebGPUTensor& t_in = graph.get_tensor(r.in_id);
  const WebGPUTensor& t_w = graph.get_tensor(r.weight_id);
  const WebGPUTensor& t_out = graph.get_tensor(r.out_id);
  const WebGPUTensor& t_resid = graph.get_tensor(resid_id);
  const WebGPUTensor& t_addout = graph.get_tensor(out_id);

  RmsAddCheck c;
  c.adjacent = true;
  c.exactly_one_operand_is_rms_out = (first_is_rms != second_is_rms);
  c.row_width = r.row_width;
  c.rms_in_numel = numel_of_tensor(t_in);
  c.rms_out_numel = numel_of_tensor(t_out);
  c.resid_numel = numel_of_tensor(t_resid);
  c.add_out_numel = numel_of_tensor(t_addout);
  c.exact_shape_match = t_in.dims == t_out.dims && t_in.dims == t_resid.dims &&
      t_in.dims == t_addout.dims;
  c.all_tensors_fp32 = utils::is_fp32_tensor(t_in) &&
      utils::is_fp32_tensor(t_w) && utils::is_fp32_tensor(t_out) &&
      utils::is_fp32_tensor(t_resid) && utils::is_fp32_tensor(t_addout);
  c.alpha = alpha;
  c.addout_is_out_buffer = (t_addout.buffer == t_out.buffer);
  c.addout_is_in_buffer = (t_addout.buffer == t_in.buffer);
  c.addout_is_weight_buffer = (t_addout.buffer == t_w.buffer);
  c.resid_is_out_buffer = (t_resid.buffer == t_out.buffer);
  if (!rms_add_fusable(c)) {
    return false;
  }
  install(
      graph,
      r.dispatch_index,
      kRmsNormVec4AddWGSL,
      {
          {0, WGPUBufferBindingType_Storage, t_out.buffer, t_out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, t_in.buffer, t_in.nbytes},
          {2, WGPUBufferBindingType_ReadOnlyStorage, t_w.buffer, t_w.nbytes},
          {3, WGPUBufferBindingType_Uniform, r.params_buffer, 16u},
          {4,
           WGPUBufferBindingType_ReadOnlyStorage,
           t_resid.buffer,
           t_resid.nbytes},
          {5, WGPUBufferBindingType_Storage, t_addout.buffer, t_addout.nbytes},
      });

  // The rms_norm resize hook already rewrites params_buf (unchanged 16-byte
  // layout) and workgroup_count_x, and sets cur_dims(rms out). The add's own
  // hooks are never registered, so mirror its output extent here. Registered
  // after the rms hook, so it runs after cur_dims(rms out) has converged.
  const int rms_in_id = r.in_id;
  const int fused_resid_id = resid_id;
  const int addout_id = out_id;
  auto resize_add = [rms_in_id, fused_resid_id, addout_id](WebGPUGraph& g) {
    resize_fused_add_outputs(g, rms_in_id, fused_resid_id, addout_id);
  };
  graph.add_tensor_resize_hook(rms_in_id, resize_add);
  if (fused_resid_id != rms_in_id) {
    graph.add_tensor_resize_hook(fused_resid_id, resize_add);
  }

  r.add_fused = true;
  r.resid_id = resid_id;
  r.addout_id = out_id;
  graph.offer_rms_fusion_site(std::move(r));
  return true;
}

bool try_fuse_scale(WebGPUGraph& graph, int in1_id, int in2_id, int out_id) {
  Record r = graph.rms_fusion_site();
  if (!still_last(graph, r) || !r.add_fused) {
    return false;
  }

  const bool first_is_add = (in1_id == r.addout_id);
  const bool second_is_add = (in2_id == r.addout_id);
  const int scale_id = first_is_add ? in2_id : in1_id;

  const WebGPUTensor& t_in = graph.get_tensor(r.in_id);
  const WebGPUTensor& t_w = graph.get_tensor(r.weight_id);
  const WebGPUTensor& t_out = graph.get_tensor(r.out_id);
  const WebGPUTensor& t_resid = graph.get_tensor(r.resid_id);
  const WebGPUTensor& t_addout = graph.get_tensor(r.addout_id);
  const WebGPUTensor& t_scale = graph.get_tensor(scale_id);
  const WebGPUTensor& t_scaleout = graph.get_tensor(out_id);

  RmsScaleCheck c;
  c.adjacent_fused_add = true;
  c.exactly_one_operand_is_add_out = (first_is_add != second_is_add);
  c.scale_numel = numel_of_tensor(t_scale);
  c.add_out_numel = numel_of_tensor(t_addout);
  c.mul_out_numel = numel_of_tensor(t_scaleout);
  c.all_tensors_fp32 = utils::is_fp32_tensor(t_in) &&
      utils::is_fp32_tensor(t_w) && utils::is_fp32_tensor(t_out) &&
      utils::is_fp32_tensor(t_resid) && utils::is_fp32_tensor(t_addout) &&
      utils::is_fp32_tensor(t_scale) && utils::is_fp32_tensor(t_scaleout);
  c.scaleout_is_out_buffer = (t_scaleout.buffer == t_out.buffer);
  c.scaleout_is_in_buffer = (t_scaleout.buffer == t_in.buffer);
  c.scaleout_is_weight_buffer = (t_scaleout.buffer == t_w.buffer);
  c.scaleout_is_addout_buffer = (t_scaleout.buffer == t_addout.buffer);
  c.scaleout_is_resid_buffer = (t_scaleout.buffer == t_resid.buffer);
  c.scale_is_out_buffer = (t_scale.buffer == t_out.buffer);
  c.scale_is_addout_buffer = (t_scale.buffer == t_addout.buffer);
  if (!rms_scale_fusable(c)) {
    return false;
  }
  install(
      graph,
      r.dispatch_index,
      kRmsNormVec4AddScaleWGSL,
      {
          {0, WGPUBufferBindingType_Storage, t_out.buffer, t_out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, t_in.buffer, t_in.nbytes},
          {2, WGPUBufferBindingType_ReadOnlyStorage, t_w.buffer, t_w.nbytes},
          {3, WGPUBufferBindingType_Uniform, r.params_buffer, 16u},
          {4,
           WGPUBufferBindingType_ReadOnlyStorage,
           t_resid.buffer,
           t_resid.nbytes},
          {5, WGPUBufferBindingType_Storage, t_addout.buffer, t_addout.nbytes},
          {6,
           WGPUBufferBindingType_ReadOnlyStorage,
           t_scale.buffer,
           t_scale.nbytes},
          {7,
           WGPUBufferBindingType_Storage,
           t_scaleout.buffer,
           t_scaleout.nbytes},
      });

  const int rms_in_id = r.in_id;
  const int resid_id = r.resid_id;
  const int addout_id = r.addout_id;
  const int fused_scale_id = scale_id;
  const int scaleout_id = out_id;
  auto resize_scale =
      [rms_in_id, resid_id, addout_id, fused_scale_id, scaleout_id](
          WebGPUGraph& g) {
        resize_fused_add_outputs(
            g, rms_in_id, resid_id, addout_id, fused_scale_id, scaleout_id);
      };
  graph.add_tensor_resize_hook(rms_in_id, resize_scale);
  if (resid_id != rms_in_id) {
    graph.add_tensor_resize_hook(resid_id, resize_scale);
  }
  if (fused_scale_id != rms_in_id && fused_scale_id != resid_id) {
    graph.add_tensor_resize_hook(fused_scale_id, resize_scale);
  }

  // Nothing further can chain onto this dispatch.
  graph.clear_rms_fusion_site();
  return true;
}

} // namespace fusion
} // namespace executorch::backends::webgpu
