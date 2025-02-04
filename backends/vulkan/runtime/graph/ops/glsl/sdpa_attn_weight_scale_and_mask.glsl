/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "rw", "attn_weight", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "ivec4", "attn_weight_sizes")}
  ${layout_declare_ubo(B, "ivec4", "attn_weight_strides")}
$else:
  ${layout_declare_ubo(B, "ivec3", "attn_weight_limits")}

${layout_declare_ubo(B, "int", "input_pos")}
${layout_declare_ubo(B, "float", "scale")}


#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Negative infinity is represented by having sign bit be 1, all exponent bits
// be 1, all mantissa bits be 0.
#define NEGATIVE_INF_BITS 0xFF800000
const float negative_infinity = NEGATIVE_INF_BITS;

#ifdef USING_BUFFER

/*
 * This implementations applies a scale and mask to the attention weight tensor
 * of an SDPA block. The sizes of the attention weight is
 * (batch_size, n_heads, seq_len, input_pos + seq_len)
 * Conceptually the weights represent the relationship between each token in the
 * sequence with each token preceding it.
 *
 * The scale applied is 1.0 / sqrt(head_dim_length)
 *
 * The mask applied is a bit more complicated. Imagine you create a square
 * matrix of size (input_pos + seq_len, input_pos + seq_len), and then set the
 * lower triangular section of the matrix to -inf. Then, slice the matrix along
 * the row dimension starting from input_pos to input_pos + seq_len. You end up
 * with a partial mask with size (seq_len, input_pos + seq_len). This is the
 * mask that is applied to the attention weight.
 *
 * In the shader, instead of generating the mask, the index of the elment is
 * inspected to determine if it would have been masked. Given an element at
 * tensor index (n, c, h, w), it would be masked if w < h + input_pos.
 */

/***************************
 ** Buffer Implementation **
 ***************************/

void main() {
  const ivec4 attn_weight_idx = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z,
      0);

  if (any(greaterThanEqual(attn_weight_idx, attn_weight_sizes))) {
    return;
  }

  const T scale_conv = T(scale);

  const int attn_weight_id = tidx_to_bufi(attn_weight_idx, attn_weight_strides);
  if (attn_weight_idx.x <= attn_weight_idx.y + input_pos) {
    attn_weight[attn_weight_id] = attn_weight[attn_weight_id] * scale_conv;
  } else {
    attn_weight[attn_weight_id] = T(negative_infinity);
  }
}

#else

/****************************
 ** Texture Implementation **
 ****************************/

/*
 * This implementation assumes that the attention weight is width packed, i.e.
 * the packed dim of the attn_weight is 0.
 */
void main() {
  const ivec3 attn_weight_pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(attn_weight_pos, attn_weight_limits))) {
    return;
  }

  vec4 outtex = imageLoad(attn_weight, attn_weight_pos) * scale;

  // Mask out the upper triangular of attn_weight to -inf
  [[unroll]] for (int i = 0; i < 4; ++i) {
    if (attn_weight_pos.x * 4 + i > attn_weight_pos.y + input_pos) {
      outtex[i] = negative_infinity;
    }
  }

  write_texel(attn_weight, attn_weight_pos, outtex);
}

#endif // USING_BUFFER
