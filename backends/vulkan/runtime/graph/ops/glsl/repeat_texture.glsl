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

${define_active_storage_type("texture3d")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "out_meta")}
${layout_declare_ubo(B, "TextureMetadata", "in_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, out_meta)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(out_meta, out_pos);

  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, out_meta.sizes[out_meta.packed_dim] - out_tidx.data[out_meta.packed_dim]);
  for (int comp = 0; comp < 4; comp++) {
    TensorIndex4D in_tidx = out_tidx;
    in_tidx.data = ivec4(
        out_tidx.data.x % in_meta.sizes.x,
        out_tidx.data.y % in_meta.sizes.y,
        out_tidx.data.z % in_meta.sizes.z,
        out_tidx.data.w % in_meta.sizes.w);

    TextureElementIndex in_elem =
        tensor4d_idx_to_texture_element_idx_simple(in_meta, in_tidx);

    VEC4_T in_texel = texelFetch(t_in, in_elem.pos, 0);
    out_texel[comp] = in_texel[in_elem.comp];

    out_tidx.data[out_meta.packed_dim]++;
  }

  imageStore(t_out, out_pos, out_texel);
}
