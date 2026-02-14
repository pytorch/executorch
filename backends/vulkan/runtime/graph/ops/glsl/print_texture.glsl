/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#extension GL_EXT_debug_printf : enable

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#define DEBUG_MODE

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "r", "t_inp", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(push_constant) uniform restrict Block {
  int value_ref;
};

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(pos, inp)) {
    return;
  }

  TensorIndex4D tidx = texture_pos_to_tensor4d_idx_simple(inp, pos);
  VEC4_T texel = texelFetch(t_inp, pos, 0);

  int limit = min(
      4, inp.sizes[inp.packed_dim] - tidx.data[inp.packed_dim]);
  for (int i = 0; i < limit; i++) {
    float v = float(texel[i]);
    if (abs(v) > 1e5) {
      ivec4 idx = tidx.data;
      idx[inp.packed_dim] += i;
      debugPrintfEXT(
          "[print_texture] value_ref=%d, sizes=(%d, %d, %d, %d), idx=(%d, %d, %d, %d), value=%f\\n",
          value_ref,
          inp.sizes.x, inp.sizes.y, inp.sizes.z, inp.sizes.w,
          idx.x, idx.y, idx.z, idx.w,
          v);
    }
  }
}
