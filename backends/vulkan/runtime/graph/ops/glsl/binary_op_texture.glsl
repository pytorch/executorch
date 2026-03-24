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

${define_required_extensions("texture3d", DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions("texture3d", "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
$if IS_COMPARISON_OP:
  #define T ${texel_load_component_type("uint8", "texture3d")}
  #define VEC4_OUT_T ${texel_load_type("uint8", "texture3d")}
$else:
  #define T ${texel_load_component_type(DTYPE, "texture3d")}
  #define VEC4_OUT_T VEC4_T

#define op(X, Y, A) ${OPERATOR}

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

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  const int out_packed_dim = get_packed_dim(out_layout);
  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  VEC4_OUT_T outtex = VEC4_OUT_T(0);

  // Use safe_idx/safe_set to avoid dynamic UBO ivec4 indexing, which crashes
  // the Adreno 740 SPIR-V compiler when the index is a specialization constant.
  int limit = min(
      4,
      safe_idx(outp.sizes, out_packed_dim) -
          safe_idx(out_tidx.data, out_packed_dim));
  for (int comp = 0; comp < limit; comp++) {
    TensorIndex4D in_tidx;
    in_tidx.data = min(out_tidx.data, inp.sizes - 1);
    TextureElementIndex in_elem =
        tensor4d_idx_to_texture_element_idx_simple(
            inp, in_tidx, in_layout);
    VEC4_T in_texel = texelFetch(t_in, in_elem.pos, 0);

    TensorIndex4D other_tidx;
    other_tidx.data = min(out_tidx.data, otherp.sizes - 1);
    TextureElementIndex other_elem =
        tensor4d_idx_to_texture_element_idx_simple(
            otherp, other_tidx, other_layout);
    VEC4_T other_texel = texelFetch(t_other, other_elem.pos, 0);

    outtex[comp] = T(op(
        in_texel[in_elem.comp],
        other_texel[other_elem.comp],
        alpha));

    safe_set(
        out_tidx.data,
        out_packed_dim,
        safe_idx(out_tidx.data, out_packed_dim) + 1);
  }

  imageStore(t_out, out_pos, outtex);
}
