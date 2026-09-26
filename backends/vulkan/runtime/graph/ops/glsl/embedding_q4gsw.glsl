/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

$if STORAGE == "buffer":
  #define OUTPUT_BUFFER

$if IS_LINEAR_WEIGHT:
  #define LINEAR_WEIGHT
  $if WEIGHT_STORAGE == "buffer":
    #define WEIGHT_BUFFER

${define_required_extensions(STORAGE, DTYPE)}
$if IS_LINEAR_WEIGHT:
  ${define_required_extensions(WEIGHT_STORAGE, "int")}
$else:
  ${define_required_extensions("buffer", "int")}
${define_required_extensions("buffer", [SCALES_DTYPE, "uint8"])}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

// Output uses the graph's storage type
${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
// Indices use the graph's storage type
${layout_declare_tensor(B, "r", "t_indices", "int", STORAGE)}
// Weight: flat buffer for regular, packed block format for linear_weight
$if IS_LINEAR_WEIGHT:
  ${layout_declare_tensor(B, "r", "t_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "r", "t_weight", "int", "buffer", is_scalar_array=False)}
// Scales are ALWAYS buffer, loaded as scalar
${layout_declare_tensor(B, "r", "t_scales", SCALES_DTYPE, "buffer")}

// Output sizes in WHCN order
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform PushConstants {
  int group_size;
  int is_linear_weight;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Load 4 dequantized embedding values at the given dimension offset.
// The weight storage format differs between regular and linear_weight variants,
// so this function is defined separately for each.
#ifdef LINEAR_WEIGHT

// Linear-packed block format: weight is stored as blocks indexed by
// t_weight[n8 * K4 + k4], where each ivec4 element contains 8 interleaved
// 4-bit values from 2 sub-rows of an 8-row block.
VEC4_T load_embedding_weights(
    const int embedding_idx,
    const int dim,
    const int embed_dim,
    const float scale) {
  const int n8 = embedding_idx >> 3;
  const int n_local = embedding_idx & 7;
  const int row_in_block = n_local < 4 ? n_local : n_local - 4;
  const int shift_base = n_local < 4 ? 0 : 4;
  const int K4 = embed_dim >> 2;
  const int k4 = dim >> 2;

#ifdef WEIGHT_BUFFER
  const ivec4 block = t_weight[n8 * K4 + k4];
#else
  const ivec4 block = texelFetch(t_weight, ivec2(k4, n8), 0);
#endif

  const uint packed_uint = uint(block[row_in_block]);

  return VEC4_T(
      T(float(int((packed_uint >> shift_base) & 0xFu) - 8) * scale),
      T(float(int((packed_uint >> (shift_base + 8)) & 0xFu) - 8) * scale),
      T(float(int((packed_uint >> (shift_base + 16)) & 0xFu) - 8) * scale),
      T(float(int((packed_uint >> (shift_base + 24)) & 0xFu) - 8) * scale));
}

#else // !LINEAR_WEIGHT

// Flat buffer format: weight is stored as a contiguous array of packed bytes.
// Each ivec4 covers 32 int4 values (16 bytes) for one embedding row.
VEC4_T load_embedding_weights(
    const int embedding_idx,
    const int dim,
    const int embed_dim,
    const float scale) {
  const int blocks_per_row = embed_dim >> 5;
  const int block_in_row = dim >> 5;
  const int t = (dim >> 2) & 7;

  const ivec4 packed = t_weight[embedding_idx * blocks_per_row + block_in_row];
  const int int_idx = t >> 1;
  const int byte_pair = t & 1;

  const uint u = uint(packed[int_idx]);
  const int shift = byte_pair << 4;
  const uint b0 = (u >> shift) & 0xFFu;
  const uint b1 = (u >> (shift + 8)) & 0xFFu;

  return VEC4_T(
      T(float(int(b0 >> 4) - 8) * scale),
      T(float(int(b0 & 0xFu) - 8) * scale),
      T(float(int(b1 >> 4) - 8) * scale),
      T(float(int(b1 & 0xFu) - 8) * scale));
}

#endif // LINEAR_WEIGHT

void main() {
  const int block_in_row = int(gl_GlobalInvocationID.x);
  const int y_idx = int(gl_GlobalInvocationID.y);
  const int z_idx = int(gl_GlobalInvocationID.z);

  // out_sizes is in WHCN order: x=W(embed_dim), y=H, z=C, w=N
  const int embed_dim = out_sizes.x;
  const int blocks_per_row = embed_dim >> 5;
  const int out_height = out_sizes.y;
  const int num_indices = out_sizes.y * out_sizes.z * out_sizes.w;

  const int indices_idx = z_idx * out_height + y_idx;
  if (block_in_row >= blocks_per_row || indices_idx >= num_indices) {
    return;
  }

#ifdef OUTPUT_BUFFER
  const int embedding_idx = t_indices[indices_idx];
#else
  const ivec4 in_texel =
      texelFetch(t_indices, ivec3(indices_idx >> 2, 0, 0), 0);
  const int embedding_idx = in_texel[indices_idx & 3];
#endif

  const int base_dim = block_in_row << 5;
  const int groups_per_row = embed_dim / group_size;

  [[unroll]] for (int t = 0; t < 8; t++) {
    const int dim = base_dim + (t << 2);
    const float scale =
        float(t_scales[embedding_idx * groups_per_row + dim / group_size]);

    const VEC4_T vals =
        load_embedding_weights(embedding_idx, dim, embed_dim, scale);

#ifdef OUTPUT_BUFFER
    const int out_base = indices_idx * embed_dim + dim;
    t_out[out_base] = vals.x;
    t_out[out_base + 1] = vals.y;
    t_out[out_base + 2] = vals.z;
    t_out[out_base + 3] = vals.w;
#else
    imageStore(
        t_out,
        ivec3((base_dim >> 2) + t, y_idx, z_idx),
        vals);
#endif
  }
}
