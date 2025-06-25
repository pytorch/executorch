/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_mean", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_rstd", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec3", "weight_limits")}
${layout_declare_ubo(B, "ivec4", "mean_strides")}

layout(push_constant) uniform PRECISION restrict Block {
  int group;
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Applies group normalization to t_in, and write the results to t_out. The mean
 * and rstd of the input tensor are precomputed and passed in as t_mean and
 * t_rstd.
 *
 * Given an input tensor t_in of shape [N, C, H, W], the mean and rstd will have
 * shape [N, C / ngroup], and the output will have the same shape as t_in. The
 * weight and bias tensor will have a shape of [C].
 *
 * In this implementation, the input and output tensors are assumed to be
 * channels packed textures with standard axis mapping.
 *
 * The weight and bias tensors are assumed to be width packed textures with
 * standard axis mapping.
 *
 * The mean and rstd tensors are assumed to be contiguous buffer-backed tensors.
 */
void apply_group_norm() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Check bounds
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  // Convert texture position to tensor coordinates using default axis mapping
  // and channels packing
  ivec4 out_tidx = ivec4(pos.x, pos.y, mul4(pos.z), 0);

  // Handle batch dimension if batches > 1
  if (out_sizes.w > 1) {
    const int C_aligned = alignup4(out_sizes.z);
    // default axis mapping means channels is the batch concatenation dim
    const int batch_idx = out_tidx.z / C_aligned;
    out_tidx.w = batch_idx;
    out_tidx.z = out_tidx.z % C_aligned;
  }

  // Load input texel (contains 4 consecutive channels)
  const vec4 input_texel = load_texel(t_in, pos);

  // Load weight and bias texels, which are width-packed; each element along the
  // width dim corresponds to a channel in the input tensor.
  const ivec3 weight_pos = ivec3(out_tidx.z / 4, 0, 0);
  const vec4 weight_texel = load_texel(t_weight, weight_pos);
  const vec4 bias_texel = load_texel(t_bias, weight_pos);

  // Calculate which channels this texel represents
  // For channels-packed layout: texel at position z contains channels [z, z+1, z+2, z+3]
  const int base_channel = out_tidx.z;

  // Calculate buffer indices for mean/rstd lookup
  // Mean/rstd tensors have shape [G, N] where G = C/group
  const int batch_idx = out_tidx.w;
  const int channels_per_group = out_sizes.z / group;

  vec4 bias;
  // Process each element of the output texel individually, since each element
  // may belong to a different channel group
  for (int i = 0; i < 4; ++i) {
    const int channel_idx = base_channel + i;
    // Handle case where padding channels are added
    if (channel_idx >= out_sizes.z) {
      bias[i] = input_texel[i];
      continue;
    }

    // Calculate group index for this channel
    const int group_idx = channel_idx / channels_per_group;

    // Create tensor index for mean/rstd buffer access
    const ivec4 mean_tidx = ivec4(group_idx, batch_idx, 0, 0);
    const int mean_bufi = tidx_to_bufi(mean_tidx, mean_strides);

    // Load mean and rstd values for this channel
    const float mean_val = t_mean[mean_bufi];
    const float rstd_val = t_rstd[mean_bufi];

    // Apply group normalization with weight and bias: ((input - mean) * rstd) * weight + bias
    const float normalized = (input_texel[i] - mean_val) * rstd_val;
    bias[i] = normalized * weight_texel[i] + bias_texel[i];
  }

  // Write result to output texture
  write_texel(t_out, pos, bias);
}

void main() {
  apply_group_norm();
}
