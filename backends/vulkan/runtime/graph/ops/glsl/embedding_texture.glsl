/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

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

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "indices_layout", "CONTIG_LAYOUT_INT")}

int load_embedding_idx(const TensorIndex4D out_tidx) {
  TensorIndex4D indices_tidx;
  indices_tidx.data.xyz = out_tidx.data.yzw;
  indices_tidx.data.w = 0;

  TextureElementIndex elem_pos = tensor4d_idx_to_texture_element_idx_simple(
    indices, indices_tidx, indices_layout);

  const ivec4 in_texel = texelFetch(t_indices, elem_pos.pos, 0);
  return in_texel[elem_pos.comp];
}

T load_weight_element(const int embedding_idx, const int dim_idx) {
  return T(t_weight[embedding_idx * int(width(weight)) + dim_idx]);
}

// Loads 4 consecutive elements of the embedding dim from a single weight row.
// Only valid when the output texel packs along the embedding (width) dim.
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

  const TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  const int packed_dim = get_packed_dim(out_layout);

  VEC4_T weight_texel = VEC4_T(0);
  if (packed_dim == 0) {
    // The texel packs 4 consecutive elements of the embedding dim, so every
    // component reads from the same weight row.
    const int embedding_idx = load_embedding_idx(out_tidx);
    weight_texel = load_weight_texel(embedding_idx, out_tidx.data.x);
  } else {
    // The texel packs 4 consecutive elements of an index dim, so every
    // component reads a different weight row at the same embedding dim.
    const int packed_dim_limit = safe_idx(outp.sizes, packed_dim);
    const int packed_dim_start = safe_idx(out_tidx.data, packed_dim);
    const int dim_idx = out_tidx.data.x;

    TensorIndex4D lane_tidx = out_tidx;
    [[unroll]] for (int i = 0; i < 4; ++i) {
      const int packed_idx = packed_dim_start + i;
      if (packed_idx < packed_dim_limit) {
        safe_set(lane_tidx.data, packed_dim, packed_idx);
        weight_texel[i] =
            load_weight_element(load_embedding_idx(lane_tidx), dim_idx);
      }
    }
  }

  imageStore(t_out, out_pos, weight_texel);
}
