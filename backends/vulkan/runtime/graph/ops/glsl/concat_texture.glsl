/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

#define USING_TEXTURE3D

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_in" + str(i + 1), DTYPE, "texture3d")}

${layout_declare_ubo(B, "int", "concat_dim")}

$in_metadata = ""
$for i in range(NUM_INPUTS):
  $in_metadata += "ivec4 in" + str(i + 1) + "_sizes;\n"

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ${in_metadata}
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

$for i in range(NUM_INPUTS):
  ${layout_declare_spec_const(C, "int", "in" + str(i+1) + "_layout", "DEFAULT_LAYOUT")}
  const lowp ivec4 in${i+1}_axis_map = unhash_axis_map(in${i+1}_layout);
  const lowp int in${i+1}_packed_dim = unhash_packed_dim(in${i+1}_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Check if we can use the fast path (no texel merging required)
bool can_use_fast_path() {
  // Fast path is possible when:
  // 1. The concat dimension is not the packed dimension, or
  // 2. The concat dimension is the packed dimension but both input tensors have dimensions
  //    that are multiples of 4 along the packed dimension
  if (concat_dim != out_packed_dim) {
    return true;
  }

  // Check if all input tensors have dimensions that are multiples of 4 along the packed dimension
  bool all_concat_dim_size_multiple_of_4 = true;
  $for i in range(NUM_INPUTS):
    all_concat_dim_size_multiple_of_4 =
        all_concat_dim_size_multiple_of_4 &&
        (in${i+1}_sizes[concat_dim] % 4 == 0);

  return all_concat_dim_size_multiple_of_4;
}

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  ivec4 out_tidx = lpos_to_tidx(lpos, out_sizes, out_axis_map.w, out_packed_dim);

  if (any(greaterThanEqual(out_tidx, out_sizes))) {
    return;
  }

  if (can_use_fast_path()) {
    // Fast path: No texel merging required
    ivec4 in_tidx = out_tidx;

    $for i in range(NUM_INPUTS):
      // For each input tensor, check if the tensor index is within bounds. If
      // so, read the texel from the input tensor and write it to the output
      if (in_tidx[concat_dim] < in${i+1}_sizes[concat_dim]) {
        const ivec3 in_pos = tidx_to_pos(in_tidx, in${i+1}_sizes, in${i+1}_axis_map, in${i+1}_packed_dim);
        const VEC4_T in_texel = load_texel(t_in${i+1}, in_pos);
        write_texel_lpos(t_out, lpos, in_texel, out_axis_map);
        return;
      }
      // Otherwise, adjust the index along the concat dimension and try the next
      // input tensor.
      else {
        in_tidx[concat_dim] -= in${i+1}_sizes[concat_dim];
      }
  }
  else {
    // Slow path: Texel merging required
    VEC4_T out_texel = VEC4_T(0);

    // Process each element in the output texel individually
    for (int texel_i = 0; texel_i < 4; ++texel_i) {
      ivec4 curr_out_tidx = out_tidx;
      curr_out_tidx[out_packed_dim] += texel_i;

      // Skip if we're out of bounds
      if (curr_out_tidx[out_packed_dim] >= out_sizes[out_packed_dim]) {
        continue;
      }

      ivec4 in_tidx = curr_out_tidx;
      $for i in range(NUM_INPUTS):
        // For each input tensor, check if the tensor index is within bounds. If
        // so, read the corresponding texel element from the input tensor and
        // write it to the output texel.
        if (in_tidx[concat_dim] < in${i+1}_sizes[concat_dim]) {
          const ivec4 in_posi = tidx_to_posi(in_tidx, in${i+1}_sizes, in${i+1}_axis_map, in${i+1}_packed_dim);
          out_texel[texel_i] = load_texel(t_in${i+1}, in_posi.xyz)[in_posi.w];
          continue;
        }
        // Otherwise, adjust the index along the concat dimension and try the
        // next input tensor.
        else {
          in_tidx[concat_dim] -= in${i+1}_sizes[concat_dim];
        }
    }

    write_texel_lpos(t_out, lpos, out_texel, out_axis_map);
  }
}
