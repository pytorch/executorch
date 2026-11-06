/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

$IS_FLOAT = DTYPE in ["half", "float"]

#version 450 core

${define_required_extensions("texture3d", DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions("texture3d", "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
// Comparison ops output booleans, stored as uint8.
$if IS_COMPARISON_OP:
  #define T ${texel_load_component_type("uint8", "texture3d")}
  #define VEC4_OUT_T ${texel_load_type("uint8", "texture3d")}
$else:
  #define T ${texel_load_component_type(DTYPE, "texture3d")}
  #define VEC4_OUT_T VEC4_T

#define IN_T VEC4_T
#define OUT_T VEC4_OUT_T
#define ALPHA_T float
#define BINARY_OP_VEC4
$if IS_FLOAT:
  #define BINARY_OP_IS_FLOAT
$if IS_COMPARISON_OP:
  #define BINARY_OP_IS_COMPARISON
$elif DTYPE == "uint8":
  #define BINARY_OP_IS_BITWISE

${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing.glslh"

$if IS_COMPARISON_OP:
  ${layout_declare_tensor(B, "w", "t_out", "uint8", "texture3d")}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_other", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}
${layout_declare_ubo(B, "TextureMetadata", "otherp")}

layout(push_constant) uniform restrict Block {
  float alpha;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "other_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_broadcast_packed_dim", "0")}
${layout_declare_spec_const(C, "int", "other_broadcast_packed_dim", "0")}
${layout_declare_spec_const(C, "int", "op_type", "0")}
// Cleared when all three tensors share sizes and layout, so they share texel
// positions and no index conversion is needed.
${layout_declare_spec_const(C, "int", "should_broadcast", "1")}

#include "binary_op.glslh"

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  VEC4_T in_texel;
  VEC4_T other_texel;
  if (should_broadcast != 0) {
    TensorIndex4D out_tidx =
        texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

    // Broadcast: clamp each tensor index to input sizes
    TensorIndex4D in_tidx;
    in_tidx.data = min(out_tidx.data, inp.sizes - 1);
    ivec3 in_pos =
        tensor4d_idx_to_texel_pos_simple(inp, in_tidx, in_layout);
    in_texel = texelFetch(t_in, in_pos, 0);

    TensorIndex4D other_tidx;
    other_tidx.data = min(out_tidx.data, otherp.sizes - 1);
    ivec3 other_pos =
        tensor4d_idx_to_texel_pos_simple(otherp, other_tidx, other_layout);
    other_texel = texelFetch(t_other, other_pos, 0);

    // Handle scalar broadcasting along packed dim: when input has size 1 along
    // the packed dim, only component 0 is valid — replicate it across all lanes.
    if (in_broadcast_packed_dim > 0) {
      in_texel = VEC4_T(in_texel.x);
    }
    if (other_broadcast_packed_dim > 0) {
      other_texel = VEC4_T(other_texel.x);
    }
  } else {
    in_texel = texelFetch(t_in, out_pos, 0);
    other_texel = texelFetch(t_other, out_pos, 0);
  }

  VEC4_OUT_T outtex = apply_binary_op(in_texel, other_texel, alpha);

  // Division turns the zero padding lanes of the last texel into NaN.
  if (op_type == BINARY_OP_DIV || op_type == BINARY_OP_FLOOR_DIVIDE) {
    const int out_packed_dim = get_packed_dim(out_layout);
    const int packed_dim_size = safe_idx(outp.sizes, out_packed_dim);
    const int nspill = packed_dim_size % 4;

    if (nspill > 0) {
      const int texels_per_batch = (packed_dim_size + 3) / 4;
      int packed_pos;
      if (out_packed_dim == 0) {
        packed_pos = out_pos.x;
      } else if (out_packed_dim == 1) {
        packed_pos = out_pos.y;
      } else {
        packed_pos = out_pos.z;
      }
      const bool is_last_texel =
          (packed_pos % texels_per_batch) == (texels_per_batch - 1);

      if (is_last_texel) {
        [[unroll]] for (int i = nspill; i < 4; i++) {
          outtex[i] = 0;
        }
      }
    }
  }

  imageStore(t_out, out_pos, outtex);
}
