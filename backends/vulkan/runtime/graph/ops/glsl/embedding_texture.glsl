/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_indices", "int", "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "indices")}
${layout_declare_ubo(B, "BufferMetadata", "weight")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

int load_embedding_idx(const TensorIndex4D out_tidx) {
  TensorIndex4D indices_tidx;
  indices_tidx.data.xyz = out_tidx.data.yzw;
  indices_tidx.data.w = 0;

  TextureElementIndex elem_pos = tensor_idx_to_texture_element_idx_simple(
    indices_tidx, indices);

  const ivec4 in_texel = texelFetch(t_indices, elem_pos.pos, 0);
  return in_texel[elem_pos.comp];
}

VEC4_T load_weight_texel(const int embedding_idx, const int dim_idx) {
  int buf_i = embedding_idx * int(width(weight)) + dim_idx;
  VEC4_T weight_texel;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    weight_texel[i] = T(t_weight[buf_i++]);
  }
  return weight_texel;
}

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor_idx_simple(out_pos, outp);
  const int embedding_idx = load_embedding_idx(out_tidx);

  const VEC4_T weight_texel = load_weight_texel(embedding_idx, out_tidx.data.x);

  imageStore(t_out, out_pos, weight_texel);
}
