/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  ivec4 permute_dims;
};

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
const int out_packed_dim = get_packed_dim(out_layout);
const int in_packed_dim = get_packed_dim(in_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Convert output tensor index to input tensor index based on permutation.
// permute_dims[i] = j means output dim i comes from input dim j.
// We write: in_tidx[permute_dims.{x,y,z,w}] = out_tidx.{x,y,z,w}
// This uses literal component access on the push constant (safe) and dynamic
// indexing into the local in_tidx variable (also safe).
ivec4 out_tidx_to_in_tidx(const ivec4 out_tidx) {
  ivec4 in_tidx;
  in_tidx[permute_dims.x] = out_tidx.x;
  in_tidx[permute_dims.y] = out_tidx.y;
  in_tidx[permute_dims.z] = out_tidx.z;
  in_tidx[permute_dims.w] = out_tidx.w;
  return in_tidx;
}

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  // Check if packed dimension is preserved in the permutation. Use safe_idx
  // to avoid dynamic indexing of push constant with spec-const-derived index.
  const bool fast_path =
      safe_idx(permute_dims, out_packed_dim) == in_packed_dim;

  if (fast_path) {
    // Fast path: packed dimension is preserved, so we can copy texels directly
    ivec4 in_tidx_data = out_tidx_to_in_tidx(out_tidx.data);
    TensorIndex4D in_tidx;
    in_tidx.data = in_tidx_data;

    ivec3 in_pos =
        tensor4d_idx_to_texel_pos_simple(inp, in_tidx, in_layout);
    VEC4_T in_texel = texelFetch(t_in, in_pos, 0);

    imageStore(t_out, out_pos, in_texel);
  } else {
    // Slow path: packed dimension is not preserved, so each element of the
    // output texel may come from a different texel in the input tensor.
    VEC4_T out_texel = VEC4_T(0);

    for (int comp = 0; comp < 4; comp++) {
      ivec4 in_tidx_data = out_tidx_to_in_tidx(out_tidx.data);
      TensorIndex4D in_tidx;
      in_tidx.data = in_tidx_data;

      TextureElementIndex in_elem =
          tensor4d_idx_to_texture_element_idx_simple(inp, in_tidx, in_layout);

      VEC4_T in_texel = texelFetch(t_in, in_elem.pos, 0);
      out_texel[comp] = in_texel[in_elem.comp];

      out_tidx.data[out_packed_dim]++;
    }

    imageStore(t_out, out_pos, out_texel);
  }
}
