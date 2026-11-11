/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

$IS_FLOAT = DTYPE in ["half", "float"]
// Comparison ops output booleans, stored as uint8.
$OUT_DTYPE = "uint8" if IS_COMPARISON_OP else DTYPE

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions(STORAGE, "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define OUT_VEC4_T ${texel_load_type(OUT_DTYPE, STORAGE)}

#define IN_T ${texel_type(DTYPE)}
#define OUT_T ${texel_type(OUT_DTYPE)}
#define ALPHA_T ${texel_component_type(DTYPE)}
#define BINARY_OP_VEC4
$if IS_FLOAT:
  #define BINARY_OP_IS_FLOAT
$if IS_COMPARISON_OP:
  #define BINARY_OP_IS_COMPARISON
$elif DTYPE == "uint8":
  #define BINARY_OP_IS_BITWISE

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_other", DTYPE, STORAGE, is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
  ${layout_declare_ubo(B, "BufferMetadata", "inp")}
  ${layout_declare_ubo(B, "BufferMetadata", "otherp")}
$else:
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
${layout_declare_spec_const(C, "int", "op_type", "0")}

#include "dispatch.glslh"
#include "binary_op.glslh"

void main() {
#ifdef USING_BUFFER
  const uint out_idx = linear_idx_from_gid();
  if (out_idx >= div_4(numel(outp))) {
    return;
  }
  // Inputs with the output's sizes and strides share its texel indices.
  const bool should_broadcast =
      !are_equal(inp, outp) || !are_equal(otherp, outp);
#else
  const ivec3 out_idx = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(out_idx, outp)) {
    return;
  }
  // Inputs with the output's sizes and layout share its texel positions.
  const bool should_broadcast = in_layout != out_layout ||
      other_layout != out_layout || inp.sizes != outp.sizes ||
      otherp.sizes != outp.sizes;
#endif

  // Checked at runtime since resizing may change which inputs broadcast.
  const int packed_dim = get_packed_dim(out_layout);
  const bool in_broadcasts_packed_dim =
      size_at(inp, packed_dim) < size_at(outp, packed_dim);
  const bool other_broadcasts_packed_dim =
      size_at(otherp, packed_dim) < size_at(outp, packed_dim);

  IN_T in_texel;
  IN_T other_texel;
  if (should_broadcast) {
#ifdef USING_BUFFER
    // Clamp each tensor index to input sizes. The packed dim has stride 1 and
    // is padded to a multiple of 4, so the clamped index of the texel's first
    // element is the start of an input texel.
    const TensorIndex out_tidx =
        linear_idx_to_tensor_idx(outp, mul_4(out_idx));

    TensorIndex in_tidx = out_tidx;
    clamp_tensor_idx(inp, in_tidx);
    in_texel = IN_T(t_in[div_4(tensor_idx_to_linear_idx(inp, in_tidx))]);

    TensorIndex other_tidx = out_tidx;
    clamp_tensor_idx(otherp, other_tidx);
    other_texel =
        IN_T(t_other[div_4(tensor_idx_to_linear_idx(otherp, other_tidx))]);
#else
    TensorIndex4D out_tidx =
        texture_pos_to_tensor4d_idx_simple(outp, out_idx, out_layout);

    // Clamp each tensor index to input sizes
    TensorIndex4D in_tidx;
    in_tidx.data = min(out_tidx.data, inp.sizes - 1);
    ivec3 in_pos =
        tensor4d_idx_to_texel_pos_simple(inp, in_tidx, in_layout);
    in_texel = IN_T(texelFetch(t_in, in_pos, 0));

    TensorIndex4D other_tidx;
    other_tidx.data = min(out_tidx.data, otherp.sizes - 1);
    ivec3 other_pos =
        tensor4d_idx_to_texel_pos_simple(otherp, other_tidx, other_layout);
    other_texel = IN_T(texelFetch(t_other, other_pos, 0));
#endif

    // Handle scalar broadcasting along packed dim: when input has size 1 along
    // the packed dim, only component 0 is valid — replicate it across all lanes.
    if (in_broadcasts_packed_dim) {
      in_texel = IN_T(in_texel.x);
    }
    if (other_broadcasts_packed_dim) {
      other_texel = IN_T(other_texel.x);
    }
  } else {
#ifdef USING_BUFFER
    in_texel = IN_T(t_in[out_idx]);
    other_texel = IN_T(t_other[out_idx]);
#else
    in_texel = IN_T(texelFetch(t_in, out_idx, 0));
    other_texel = IN_T(texelFetch(t_other, out_idx, 0));
#endif
  }

  OUT_T out_texel =
      apply_binary_op(op_type, in_texel, other_texel, ALPHA_T(alpha));

  // Replicated lanes fill the padding lanes of an input texel.
  if (!binary_op_preserves_zero(op_type, alpha) || in_broadcasts_packed_dim ||
      other_broadcasts_packed_dim) {
    out_texel = mask_padding_lanes(out_texel, out_idx, outp, packed_dim);
  }

#ifdef USING_BUFFER
  t_out[out_idx] = OUT_VEC4_T(out_texel);
#else
  imageStore(t_out, out_idx, OUT_VEC4_T(out_texel));
#endif
}
