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
${layout_declare_tensor(B, "r", "t_self", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_index", "int", "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}
${layout_declare_ubo(B, "TextureMetadata", "index")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Implements aten.index.Tensor for the case where self is 1D and there is
// exactly one index tensor. Each output element is:
//   output[...] = self[index[...]]

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(outp, out_pos);
  ivec4 idx_texel = texelFetch(t_index, out_pos, 0);

  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, outp.sizes[outp.packed_dim] - out_tidx.data[outp.packed_dim]);
  for (int comp = 0; comp < limit; comp++) {
    int idx = idx_texel[comp];

    // Construct a tensor index for the 1D self tensor.
    // In WHCN ordering, a 1D tensor has its elements along dim 0 (width).
    TensorIndex4D self_tidx;
    self_tidx.data = ivec4(idx, 0, 0, 0);

    TextureElementIndex self_elem =
        tensor4d_idx_to_texture_element_idx_simple(inp, self_tidx);

    VEC4_T self_texel = texelFetch(t_self, self_elem.pos, 0);
    out_texel[comp] = self_texel[self_elem.comp];

    out_tidx.data[outp.packed_dim]++;
  }

  imageStore(t_out, out_pos, out_texel);
}
