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
#include <executorch/backends/webgpu/runtime/ops/slice/SliceDispatch.h>
#include <executorch/backends/webgpu/runtime/ops/slice/slice_dual_guard.h>
#include <executorch/backends/webgpu/runtime/ops/slice/slice_dual_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/slice/slice_wgsl.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct SliceParams {
  uint32_t dim;
  uint32_t start;
  uint32_t step;
  uint32_t _pad;
};

// Read scalar arg: Int->value (INT64_MAX->default), Double->truncated int if
// integral (the edge dialect may serialize an integer index as a float, e.g.
// a 0 start; a fractional Double throws, it is not a valid index),
// Null->default, else throw.
int64_t
read_scalar(WebGPUGraph& graph, int id, int64_t dflt, const char* what) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::Int: {
      const int64_t v = graph.get_int(id);
      return v == INT64_MAX ? dflt : v;
    }
    case WebGPUGraph::ValueType::Double: {
      const double d = graph.get_double(id);
      // Casting a NaN or out-of-int64-range double is undefined behavior;
      // reject before the cast, not after.
      if (std::isnan(d) || d < -9223372036854775808.0 ||
          d >= 9223372036854775808.0) {
        throw std::runtime_error(std::string("slice: non-integral ") + what);
      }
      const int64_t v = static_cast<int64_t>(d);
      if (static_cast<double>(v) != d) {
        throw std::runtime_error(std::string("slice: non-integral ") + what);
      }
      return v;
    }
    case WebGPUGraph::ValueType::Null:
      return dflt;
    default:
      throw std::runtime_error(
          std::string("slice: dynamic/unsupported ") + what);
  }
}

// Read a slice index (start/end) that MAY be a dynamic SymInt; else Int/Double
// (truncated int if integral, mirrors read_scalar)/Null.
int64_t read_index(WebGPUGraph& graph, int id, int64_t dflt) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::SymInt:
      return graph.read_symint(id);
    case WebGPUGraph::ValueType::Int: {
      const int64_t v = graph.get_int(id);
      return v == INT64_MAX ? dflt : v;
    }
    case WebGPUGraph::ValueType::Double: {
      const double d = graph.get_double(id);
      // Casting a NaN or out-of-int64-range double is undefined behavior;
      // reject before the cast, not after.
      if (std::isnan(d) || d < -9223372036854775808.0 ||
          d >= 9223372036854775808.0) {
        throw std::runtime_error("slice: non-integral start/end index");
      }
      const int64_t v = static_cast<int64_t>(d);
      if (static_cast<double>(v) != d) {
        throw std::runtime_error("slice: non-integral start/end index");
      }
      return v;
    }
    case WebGPUGraph::ValueType::Null:
      return dflt;
    default:
      throw std::runtime_error("slice: dynamic/unsupported start/end index");
  }
}

bool is_symint(WebGPUGraph& graph, int id) {
  return graph.get_value_type(id) == WebGPUGraph::ValueType::SymInt;
}

// Clamp + normalize a (possibly negative) index into [0, size].
int64_t norm_clamp(int64_t idx, int64_t size) {
  if (idx < 0) {
    idx += size;
  }
  return idx < 0 ? 0 : (idx > size ? size : idx);
}

// ---------------------------------------------------------------------------
// Dual-store dispatch merge.
//
// In the sealed Gemma4 decode graph, 67 of the 250 aten.slice_copy.Tensor calls
// take the output of the IMMEDIATELY preceding slice_copy and re-slice it over
// its whole extent -- a pure elementwise copy, 126 of the 250 slices are such
// full-span copies overall. Instead of a second gather dispatch, the preceding
// slice's dispatch is re-bound to store its gathered value to BOTH
// destinations. The first destination is still written, so no consumer of it is
// assumed dead, nothing is reordered, and no buffer is aliased: it is a pure
// dispatch merge.
// ---------------------------------------------------------------------------

// True when read_index would return a value rather than throw. Eligibility is
// a "no" here, never a build-time throw: `end` is otherwise read only by the
// resize hook, which never runs on a static graph.
bool is_static_index(WebGPUGraph& graph, int id) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::Int:
    case WebGPUGraph::ValueType::Null:
      return true;
    case WebGPUGraph::ValueType::Double: {
      const double d = graph.get_double(id);
      return !std::isnan(d) && d >= -9223372036854775808.0 &&
          d < 9223372036854775808.0 &&
          static_cast<double>(static_cast<int64_t>(d)) == d;
    }
    default:
      return false;
  }
}

// Resolve the slice bounds at the serialized maximum shape and hand them to the
// pure slice_dual_full_span predicate (unit-tested by slice_dual_guard_test).
bool is_static_full_span(
    WebGPUGraph& graph,
    int start_id,
    int end_id,
    int64_t step,
    int64_t dim_size) {
  SliceDualSpan s;
  s.step = step;
  s.start_is_symint =
      graph.get_value_type(start_id) == WebGPUGraph::ValueType::SymInt;
  s.end_is_symint =
      graph.get_value_type(end_id) == WebGPUGraph::ValueType::SymInt;
  s.dim_size = dim_size;
  if (!s.start_is_symint) {
    if (!is_static_index(graph, start_id)) {
      return false;
    }
    s.start = read_index(graph, start_id, 0);
  }
  if (!s.end_is_symint) {
    if (!is_static_index(graph, end_id)) {
      return false;
    }
    s.end = read_index(graph, end_id, dim_size);
  }
  return slice_dual_full_span(s);
}

// Re-bind the recorded dispatch to slice_dual.wgsl with `out2` appended.
// Returns false (leaving the graph untouched) unless every guard holds.
bool try_dual_store(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    int start_id,
    int end_id,
    int64_t dim,
    int64_t step) {
  const WebGPUGraph::SliceChain& c = graph.slice_chain();
  if (!c.valid || c.out_id != in_id ||
      c.dispatch_idx + 1 != graph.num_dispatches()) {
    return false;
  }

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  // Pure copy: identical extents on every dim, full static span on the sliced
  // dim, and a distinct destination buffer from both operands of the gather.
  if (in_tensor.dims != out_tensor.dims ||
      in_tensor.nbytes != out_tensor.nbytes) {
    return false;
  }
  if (!is_static_full_span(
          graph, start_id, end_id, step, in_tensor.dims[dim])) {
    return false;
  }
  if (!slice_dual_buffers_ok(out_tensor.buffer, c.out_buffer, c.in_buffer)) {
    return false;
  }
  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  if (out_tensor.nbytes != out_numel * sizeof(float)) {
    return false;
  }

  WGPUDevice device = graph.device();

  // No override constants: slice_dual declares @workgroup_size(64, 1, 1).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kSliceDualWGSL,
      {
          {0, WGPUBufferBindingType_ReadOnlyStorage, c.in_buffer, c.in_nbytes},
          {1, WGPUBufferBindingType_Storage, c.out_buffer, c.out_nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           c.out_meta_buf,
           sizeof(TensorMeta)},
          {3, WGPUBufferBindingType_Uniform, c.in_meta_buf, sizeof(TensorMeta)},
          {4, WGPUBufferBindingType_Uniform, c.params_buf, sizeof(SliceParams)},
          {5,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
      });

  // The graph owns a dispatch's pipeline/bind group (released in its dtor), so
  // the ones being replaced are released here. bundle.pipeline/bind_group are
  // deliberately NOT released by ~ComputePipelineBundle: they move to the
  // dispatch, exactly as add_dispatch takes them everywhere else.
  WebGPUDispatch& d = graph.dispatch_at(c.dispatch_idx);
  if (d.pipeline) {
    wgpuComputePipelineRelease(d.pipeline);
  }
  if (d.bind_group) {
    wgpuBindGroupRelease(d.bind_group);
  }
  d.pipeline = bundle.pipeline;
  d.bind_group = bundle.bind_group;

  // The recorded slice's own hook already rewrites the metas/params and the
  // workgroup count and sets cur_dims(in_id). This mirrors the extent onto the
  // second destination -- the exact trigger and result the skipped slice's own
  // recompute would have produced, registered later so it runs after it.
  graph.add_tensor_resize_hook(in_id, [in_id, out_id](WebGPUGraph& g) {
    g.set_cur_dims(out_id, g.cur_dims(in_id));
  });

  // slice_dual has exactly two destinations; a third chained copy must not try
  // to attach to this dispatch.
  graph.clear_slice_chain();
  return true;
}

void slice_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, start, end, step, out]. start/end may be dynamic SymInts;
  // a resize hook recomputes the live extent on `dim` (out[dim] / cur_dims).
  const int in_id = args.at(0);
  const int start_id = args.at(2);
  const int end_id = args.at(3);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  const int in_ndim = static_cast<int>(in_tensor.dims.size());
  int64_t dim = read_scalar(graph, args.at(1), 0, "dim");
  if (dim < 0) {
    dim += in_ndim;
  }
  if (dim < 0 || dim >= in_ndim) {
    throw std::runtime_error("slice: dim out of range");
  }
  const int64_t step = read_scalar(graph, args.at(4), 1, "step");
  if (step < 1) {
    throw std::runtime_error("slice: step must be >= 1");
  }
  // start/end may be dynamic SymInts; seed from current (max) dims, the resize
  // hook recomputes live. Clamp guards the gather offset.
  const int64_t in_size = in_tensor.dims[dim];
  const int64_t start = norm_clamp(read_index(graph, start_id, 0), in_size);

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(in_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      in_tensor.nbytes != static_cast<size_t>(in_meta.numel) * sizeof(float)) {
    throw std::runtime_error("slice: non-fp32 operand (nbytes != numel * 4)");
  }

  SliceParams params = {};
  params.dim = static_cast<uint32_t>(dim);
  params.start = static_cast<uint32_t>(start);
  params.step = static_cast<uint32_t>(step);

  // Dispatch merge: when this slice is a whole-extent copy of the slice
  // emitted immediately before it, re-bind that dispatch to store into this
  // output too instead of emitting a second gather.
  if (try_dual_store(graph, in_id, out_id, start_id, end_id, dim, step)) {
    return;
  }

  uint32_t wg_size = utils::clamp_workgroup_size(device, kSliceWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "slice");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(SliceParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(SliceParams));

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
          {2, WGPUBufferBindingType_Uniform, out_meta_buf, sizeof(TensorMeta)},
          {3, WGPUBufferBindingType_Uniform, in_meta_buf, sizeof(TensorMeta)},
          {4, WGPUBufferBindingType_Uniform, params_buf, sizeof(SliceParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx =
      graph.add_dispatch({bundle.pipeline, bundle.bind_group, 1u, "slice"});
  set_slice_dispatch_grid(graph, dispatch_idx, workgroup_count);

  // Dynamic shapes: live start/end -> out[dim] len + meta/params/dispatch.
  auto recompute = [in_id,
                    out_id,
                    start_id,
                    end_id,
                    dim,
                    step,
                    wg_size,
                    out_meta_buf,
                    in_meta_buf,
                    params_buf,
                    dispatch_idx](WebGPUGraph& g) {
    const auto& in_dims = g.cur_dims(in_id);
    const int64_t live_in_size = in_dims[dim];
    const int64_t start = norm_clamp(read_index(g, start_id, 0), live_in_size);
    const int64_t end =
        norm_clamp(read_index(g, end_id, live_in_size), live_in_size);
    const int64_t len = end > start ? (end - start + step - 1) / step : 0;

    // Out dims = live input dims (mirror Vulkan resize_slice_copy_node).
    std::vector<int64_t> od = in_dims;
    od[dim] = len;
    g.set_cur_dims(out_id, od);

    WebGPUTensor t_out;
    t_out.dims = od;
    WebGPUTensor t_in;
    t_in.dims = in_dims;
    TensorMeta om;
    TensorMeta im;
    fill_tensor_meta(t_out, &om);
    fill_tensor_meta(t_in, &im);
    wgpuQueueWriteBuffer(g.queue(), out_meta_buf, 0, &om, sizeof(om));
    wgpuQueueWriteBuffer(g.queue(), in_meta_buf, 0, &im, sizeof(im));
    SliceParams p = {};
    p.dim = static_cast<uint32_t>(dim);
    p.start = static_cast<uint32_t>(start);
    p.step = static_cast<uint32_t>(step);
    wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
    const utils::WgCount wgc = utils::compute_2d_workgroup_count(
        g.device(), om.numel, wg_size, "slice(resize)");
    set_slice_dispatch_grid(g, dispatch_idx, wgc);
  };
  if (is_symint(graph, start_id)) {
    graph.add_resize_hook(start_id, recompute);
  }
  if (is_symint(graph, end_id) && end_id != start_id) {
    graph.add_resize_hook(end_id, recompute);
  }
  graph.add_tensor_resize_hook(in_id, recompute);

  // Graph owns the uniforms so the resize hook can rewrite them; freed in dtor.
  graph.own_uniform_buffer(out_meta_buf);
  graph.own_uniform_buffer(in_meta_buf);
  graph.own_uniform_buffer(params_buf);

  // Offer this dispatch to a following whole-extent copy of `out`.
  WebGPUGraph::SliceChain chain;
  chain.valid = (wg_size == kSliceDualWorkgroupSizeX);
  chain.out_id = out_id;
  chain.dispatch_idx = dispatch_idx;
  chain.in_buffer = in_tensor.buffer;
  chain.in_nbytes = in_tensor.nbytes;
  chain.out_buffer = out_tensor.buffer;
  chain.out_nbytes = out_tensor.nbytes;
  chain.out_meta_buf = out_meta_buf;
  chain.in_meta_buf = in_meta_buf;
  chain.params_buf = params_buf;
  graph.offer_slice_chain(chain);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.slice_copy.Tensor, slice_impl);
}

} // namespace executorch::backends::webgpu
