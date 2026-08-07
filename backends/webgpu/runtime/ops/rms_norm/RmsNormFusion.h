/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <webgpu/webgpu.h>

namespace executorch::backends::webgpu {

class WebGPUGraph;

namespace fusion {

// Dispatch-count reduction for the sealed Gemma4 decode round.
//
// et_vk.rms_norm.default is followed, 121 times out of 284, by an
// aten.add.Tensor that consumes its output with no dispatching op in between
// and an exactly matching shape; 43 of those adds are then followed by an
// aten.mul.Tensor against a single-element tensor. Each of those is a separate
// dispatchWorkgroups over the same 3 x 256 elements the rms_norm workgroup has
// already loaded.
//
// This folds them into the rms_norm dispatch by REWRITING the pipeline and
// bind group of the dispatch rms_norm_impl just emitted -- it never reorders,
// never drops a store, and never aliases a buffer, so it is a pure dispatch
// merge. Every intermediate (t_out, t_addout) is still written, so no consumer
// of them is assumed dead.
//
// The state machine is: record_rms_norm -> try_fuse_add -> try_fuse_scale.
// Any op that emits a dispatch in between invalidates the record, because
// every step re-checks that the recorded dispatch is still the LAST one.

// ---------------------------------------------------------------------------
// Pure fusibility predicates (no GPU, no graph): unit-tested by the guard test.
// ---------------------------------------------------------------------------

struct RmsAddCheck {
  // The recorded rms_norm dispatch is still graph.num_dispatches() - 1.
  bool adjacent = false;
  // Exactly one of the add's two tensor operands is the rms_norm output.
  bool exactly_one_operand_is_rms_out = false;
  uint32_t row_width = 0;
  uint64_t rms_in_numel = 0;
  uint64_t rms_out_numel = 0;
  uint64_t resid_numel = 0;
  uint64_t add_out_numel = 0;
  bool exact_shape_match = false;
  bool all_tensors_fp32 = false;
  float alpha = 0.0f;
  // Buffer-identity facts. The fused kernel writes t_out and t_addout and
  // reads t_in, t_weight and t_resid inside one dispatch, so any overlap that
  // the two-dispatch form made safe by the inter-dispatch barrier must reject.
  bool addout_is_out_buffer = false;
  bool addout_is_in_buffer = false;
  bool addout_is_weight_buffer = false;
  bool resid_is_out_buffer = false;
};

struct RmsScaleCheck {
  // A rms+add fusion was installed and is still the LAST dispatch.
  bool adjacent_fused_add = false;
  // Exactly one of the mul's two operands is the fused add's output.
  bool exactly_one_operand_is_add_out = false;
  uint64_t scale_numel = 0; // must be 1 (binary_mul collapses to input2[0])
  uint64_t add_out_numel = 0;
  uint64_t mul_out_numel = 0;
  bool all_tensors_fp32 = false;
  bool scaleout_is_out_buffer = false;
  bool scaleout_is_in_buffer = false;
  bool scaleout_is_weight_buffer = false;
  bool scaleout_is_addout_buffer = false;
  bool scaleout_is_resid_buffer = false;
  bool scale_is_out_buffer = false;
  bool scale_is_addout_buffer = false;
};

struct RmsResizeCheck {
  bool residual_shape_matches = false;
  bool has_scale = false;
  uint64_t scale_numel = 0;
};

// A rejected reason string (nullptr when fusible) makes the guard test able to
// assert WHICH guard fired, not just that one did. Header-inline so the guard
// test links without any GPU object.
inline const char* rms_add_reject_reason(const RmsAddCheck& c) {
  if (!c.adjacent) {
    return "rms_norm dispatch is no longer the last one";
  }
  if (!c.exactly_one_operand_is_rms_out) {
    return "add does not consume the rms_norm output exactly once";
  }
  if (c.row_width == 0u || c.row_width % 4u != 0u) {
    return "row_width is not a positive multiple of 4 (vec4 route only)";
  }
  if (c.rms_in_numel == 0u || c.rms_in_numel % c.row_width != 0u) {
    return "rms input numel is not a positive multiple of row_width";
  }
  if (c.rms_out_numel != c.rms_in_numel || c.resid_numel != c.rms_in_numel ||
      c.add_out_numel != c.rms_in_numel) {
    return "operand numels differ (the fused add is elementwise, not broadcast)";
  }
  if (!c.exact_shape_match) {
    return "operand shapes differ (the fused add does not implement broadcast)";
  }
  if (!c.all_tensors_fp32) {
    return "rms/add tensors are not all fp32";
  }
  if (c.alpha != 1.0f) {
    return "alpha != 1 (the fused kernel folds the add as a bare r + n)";
  }
  if (c.addout_is_out_buffer) {
    return "add output aliases the rms_norm output buffer";
  }
  if (c.addout_is_in_buffer) {
    return "add output aliases the rms_norm input buffer";
  }
  if (c.addout_is_weight_buffer) {
    return "add output aliases the rms_norm weight buffer";
  }
  if (c.resid_is_out_buffer) {
    return "add residual aliases the rms_norm output buffer";
  }
  return nullptr;
}

inline const char* rms_scale_reject_reason(const RmsScaleCheck& c) {
  if (!c.adjacent_fused_add) {
    return "no fused rms+add dispatch immediately precedes this mul";
  }
  if (!c.exactly_one_operand_is_add_out) {
    return "mul does not consume the fused add output exactly once";
  }
  if (c.scale_numel != 1u) {
    return "scale operand is not a single element";
  }
  if (c.mul_out_numel != c.add_out_numel) {
    return "mul output numel differs from the add output numel";
  }
  if (!c.all_tensors_fp32) {
    return "rms/add/scale tensors are not all fp32";
  }
  if (c.scaleout_is_out_buffer || c.scaleout_is_in_buffer ||
      c.scaleout_is_weight_buffer || c.scaleout_is_addout_buffer ||
      c.scaleout_is_resid_buffer) {
    return "mul output aliases a buffer the fused kernel already touches";
  }
  if (c.scale_is_out_buffer || c.scale_is_addout_buffer) {
    return "scale operand aliases a buffer the fused kernel writes";
  }
  return nullptr;
}

inline const char* rms_resize_reject_reason(const RmsResizeCheck& c) {
  if (!c.residual_shape_matches) {
    return "live residual shape requires broadcast";
  }
  if (c.has_scale && c.scale_numel != 1u) {
    return "live scale is not scalar";
  }
  return nullptr;
}

inline bool rms_add_fusable(const RmsAddCheck& c) {
  return rms_add_reject_reason(c) == nullptr;
}
inline bool rms_scale_fusable(const RmsScaleCheck& c) {
  return rms_scale_reject_reason(c) == nullptr;
}

// ---------------------------------------------------------------------------
// Build-time hooks called from the three op handlers.
// ---------------------------------------------------------------------------

// Called by rms_norm_impl right after it emits its dispatch, only for the vec4
// route (row_width % 4 == 0) at the fixed 64-wide workgroup.
void record_rms_norm(
    WebGPUGraph& graph,
    int in_id,
    int weight_id,
    int out_id,
    uint32_t num_rows,
    uint32_t row_width,
    size_t dispatch_idx,
    WGPUBuffer params_buf,
    WGPUBindGroup bind_group);

// Invalidate the record (any handler that emits a dispatch of its own).
void invalidate_record(WebGPUGraph& graph);

// Called at the top of add_impl / mul_impl. Returns true when the op has been
// folded into the preceding dispatch and the caller must emit nothing.
bool try_fuse_add(
    WebGPUGraph& graph,
    int in1_id,
    int in2_id,
    float alpha,
    int out_id);
bool try_fuse_scale(WebGPUGraph& graph, int in1_id, int in2_id, int out_id);

} // namespace fusion
} // namespace executorch::backends::webgpu
