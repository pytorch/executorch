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

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int packed_dim = get_packed_dim(out_layout);

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, outp.sizes[packed_dim] - out_tidx.data[packed_dim]);
  for (int comp = 0; comp < 4; comp++) {
    if (comp >= limit) {
      break;
    }

    // Map output tensor index to input tensor index using modulo
    TensorIndex4D inp_tidx;
    inp_tidx.data.x = out_tidx.data.x % inp.sizes.x;
    inp_tidx.data.y = out_tidx.data.y % inp.sizes.y;
    inp_tidx.data.z = out_tidx.data.z % inp.sizes.z;
    inp_tidx.data.w = out_tidx.data.w % inp.sizes.w;

    TextureElementIndex inp_elem =
        tensor4d_idx_to_texture_element_idx_simple(
            inp, inp_tidx, out_layout);

    VEC4_T inp_texel = texelFetch(t_inp, inp_elem.pos, 0);
    out_texel[comp] = inp_texel[inp_elem.comp];

    out_tidx.data[packed_dim]++;
  }

  imageStore(t_outp, out_pos, out_texel);
}
