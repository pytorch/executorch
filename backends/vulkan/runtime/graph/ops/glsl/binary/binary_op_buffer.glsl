/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

$IS_FLOAT = DTYPE in ["half", "float"]

#version 450 core

${define_required_extensions("buffer", DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions("buffer", "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define IN_T ${buffer_scalar_type(DTYPE)}
// Comparison ops output booleans, stored as uint8.
$if IS_COMPARISON_OP:
  #define T ${buffer_scalar_type("uint8")}
$else:
  #define T IN_T

#define OUT_T T
#define ALPHA_T IN_T
$if IS_FLOAT:
  #define BINARY_OP_IS_FLOAT
$if IS_COMPARISON_OP:
  #define BINARY_OP_IS_COMPARISON
$elif DTYPE == "uint8":
  #define BINARY_OP_IS_BITWISE

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

#include "dispatch.glslh"

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "other_layout", "CONTIG_LAYOUT_INT")}
// Unused by buffer path, declared for spec constant ID consistency with texture
${layout_declare_spec_const(C, "int", "in_broadcast_packed_dim", "0")}
${layout_declare_spec_const(C, "int", "other_broadcast_packed_dim", "0")}
${layout_declare_spec_const(C, "int", "op_type", "0")}
// Cleared when all three tensors share sizes and layout, so they share buffer
// indices and no index conversion is needed.
${layout_declare_spec_const(C, "int", "should_broadcast", "1")}

#include "binary_op.glslh"

void main() {
  const uint out_bufi = linear_idx_from_gid();
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  uint inp_bufi = out_bufi;
  uint other_bufi = out_bufi;
  if (should_broadcast != 0) {
    TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

    TensorIndex inp_tidx = outp_tidx;
    clamp_tensor_idx(inp, inp_tidx);

    TensorIndex other_tidx = outp_tidx;
    clamp_tensor_idx(otherp, other_tidx);

    inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);
    other_bufi = tensor_idx_to_linear_idx(otherp, other_tidx);
  }

  t_out[out_bufi] = apply_binary_op(
      op_type, t_in[inp_bufi], t_other[other_bufi], IN_T(alpha));
}
