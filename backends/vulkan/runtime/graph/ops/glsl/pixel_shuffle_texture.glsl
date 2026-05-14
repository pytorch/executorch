/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "upscale_factor", "1")}

const int out_packed_dim = get_packed_dim(out_layout);

/*
 * pixel_shuffle: rearranges (N, C*r*r, H, W) -> (N, C, H*r, W*r).
 *
 * For output element at NCHW index (n, c, h_out, w_out):
 *   w_in = w_out / r
 *   h_in = h_out / r
 *   c_in = c * r * r + (h_out % r) * r + (w_out % r)
 *
 * Each thread writes one output texel of 4 components along the packed dim.
 * Each component may map to a different input texel, so we resolve per-
 * component and use texelFetch on the input.
 */
void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  // safe_idx() avoids dynamic UBO-vector indexing, which crashes Adreno 740.
  // The output may not span a full block of 4 along the packed dim if the
  // packed-dim size is not a multiple of 4, so clamp the loop.
  const int limit = min(
      4,
      safe_idx(outp.sizes, out_packed_dim) -
          safe_idx(out_tidx.data, out_packed_dim));

  const int r = upscale_factor;

  VEC4_T out_texel = VEC4_T(0);
  for (int comp = 0; comp < 4; comp++) {
    if (comp >= limit) {
      break;
    }

    // Build the per-component output tensor index. tidx.data is a local
    // ivec4 in WHCN order ([0]=W, [1]=H, [2]=C, [3]=N), so dynamic indexing
    // here is safe (not UBO-backed).
    TensorIndex4D out_tidx_c = out_tidx;
    safe_set(
        out_tidx_c.data,
        out_packed_dim,
        safe_idx(out_tidx.data, out_packed_dim) + comp);

    const int w_out = out_tidx_c.data.x;
    const int h_out = out_tidx_c.data.y;
    const int c_out = out_tidx_c.data.z;

    const int w_in = w_out / r;
    const int h_in = h_out / r;
    const int c_in = c_out * r * r + (h_out % r) * r + (w_out % r);

    TensorIndex4D in_tidx;
    in_tidx.data = ivec4(w_in, h_in, c_in, out_tidx_c.data.w);

    TextureElementIndex in_elem =
        tensor4d_idx_to_texture_element_idx_simple(inp, in_tidx, in_layout);
    VEC4_T in_texel = texelFetch(t_in, in_elem.pos, 0);
    out_texel[comp] = in_texel[in_elem.comp];
  }

  imageStore(t_out, out_pos, out_texel);
}
