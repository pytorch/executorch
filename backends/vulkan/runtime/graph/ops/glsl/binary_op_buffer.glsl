/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Binary comparison ops require that the output is boolean and not the same as input.
$IS_COMPARISON_OP = (any([name in VARIANT_NAME for name in ["binary_eq",  "binary_lt", "binary_le", "binary_gt", "binary_ge"]]))

#version 450 core

${define_required_extensions("buffer", DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions("buffer", "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
$if IS_COMPARISON_OP:
  #define T ${buffer_scalar_type("uint8")}
  #define VEC4_OUT_T ${texel_load_type("uint8", "buffer")}
$else:
  #define T ${buffer_scalar_type(DTYPE)}
  #define VEC4_OUT_T VEC4_T

#define op(X, Y, A) ${OPERATOR}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

$if IS_COMPARISON_OP:
  ${layout_declare_tensor(B, "w", "t_out", "uint8", "buffer")}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_other", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "BufferMetadata", "otherp")}

layout(push_constant) uniform restrict Block {
  float alpha;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "other_layout", "CONTIG_LAYOUT_INT")}
// Unused by buffer path, declared for spec constant ID consistency with texture
${layout_declare_spec_const(C, "int", "in_broadcast_packed_dim", "0")}
${layout_declare_spec_const(C, "int", "other_broadcast_packed_dim", "0")}

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  // Simple case; no broadcasting
  if (are_equal(inp, otherp)) {
    t_out[out_bufi] = T(op(t_in[out_bufi], t_other[out_bufi], T(alpha)));
    return;
  }

  TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

  TensorIndex inp_tidx = outp_tidx;
  clamp_tensor_idx(inp, inp_tidx);

  TensorIndex other_tidx = outp_tidx;
  clamp_tensor_idx(otherp, other_tidx);

  uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);
  uint other_bufi = tensor_idx_to_linear_idx(otherp, other_tidx);

  t_out[out_bufi] = T(op(t_in[inp_bufi], t_other[other_bufi], T(alpha)));
}
