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

#define VEC4_T ${texel_type(DTYPE)}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  ivec4 nchw_strides = ivec4(
      1,
      outp.sizes.x,
      outp.sizes.x * outp.sizes.y,
      outp.sizes.x * outp.sizes.y * outp.sizes.z);

  int base_nchw = out_tidx.data.x * nchw_strides.x +
                  out_tidx.data.y * nchw_strides.y +
                  out_tidx.data.z * nchw_strides.z +
                  out_tidx.data.w * nchw_strides.w;

  const int out_packed_dim = get_packed_dim(out_layout);
  int packed_stride = nchw_strides[out_packed_dim];
  int limit = min(4,
      safe_idx(outp.sizes, out_packed_dim) -
      out_tidx.data[out_packed_dim]);

  VEC4_T value = VEC4_T(0);
  for (int i = 0; i < 4; i++) {
    if (i >= limit) break;

    int nchw_idx = base_nchw + i * packed_stride;

    int div_x = nchw_idx / inp.sizes.x;
    int div_y = div_x / inp.sizes.y;
    TensorIndex4D in_tidx;
    in_tidx.data = ivec4(
        nchw_idx % inp.sizes.x,
        div_x % inp.sizes.y,
        div_y % inp.sizes.z,
        div_y / inp.sizes.z);

    TextureElementIndex in_elem =
        tensor4d_idx_to_texture_element_idx_simple(inp, in_tidx, in_layout);
    VEC4_T intex = texelFetch(t_in, in_elem.pos, 0);
    value[i] = intex[in_elem.comp];
  }

  imageStore(t_out, out_pos, value);
}
