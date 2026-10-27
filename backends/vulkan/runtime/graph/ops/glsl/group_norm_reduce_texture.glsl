/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_mean", DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_rstd", DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "ivec4", "mean_strides")}
${layout_declare_ubo(B, "int", "mean_numel")}
${layout_declare_ubo(B, "ivec3", "in_limits")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(push_constant) uniform PRECISION restrict Block {
  int group;
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "mean_layout", "DEFAULT_DIM_ORDER")}
const lowp ivec4 mean_dim_order = unhash_dim_order(mean_layout);

#define LOCAL_WORK_GROUP_SIZE 64
shared float shared_sum[LOCAL_WORK_GROUP_SIZE];
shared float shared_sum_sq[LOCAL_WORK_GROUP_SIZE];

/*
 * Loads the texel at texel_idx of the range of texels this invocation's group
 * spans. mask marks the components that belong to the group; it is false for
 * components of other groups and for texels outside of the texture.
 */
vec4 load_group_texel(
    const int texel_idx,
    const int texels_in_group,
    const int start_texel_idx,
    const int group_start_channel,
    const int group_end_channel,
    const int batch_idx,
    out bvec4 mask) {
  // Convert texel index to spatial and channel coordinates
  const int spatial_idx = texel_idx / texels_in_group;
  const int texel_in_group = texel_idx % texels_in_group;

  // Convert to spatial coordinates
  const int w = spatial_idx % in_sizes.x;
  const int h = spatial_idx / in_sizes.x;

  // Calculate the global texel index
  const int global_texel_idx = start_texel_idx + texel_in_group;

  // Convert to texture position using default axis mapping
  ivec3 tex_pos = ivec3(w, h, global_texel_idx);

  // Adjust for batch dimension if needed
  if (in_sizes.w > 1) {
    // default axis mapping means channels is the batch concat dim
    tex_pos.z += batch_idx * divup4(in_sizes.z);
  }

  mask = bvec4(false);
  if (!all(lessThan(tex_pos, in_limits))) {
    return vec4(0.0);
  }

  // Mark the components of the texel that belong to this group
  const int texel_start_channel = global_texel_idx * 4;
  for (int comp = 0; comp < 4; comp++) {
    const int current_channel = texel_start_channel + comp;
    if (current_channel >= group_start_channel &&
        current_channel < group_end_channel) {
      mask[comp] = true;
    }
  }

  return vec4(load_texel(t_in, tex_pos));
}

/*
 * Computes the mean and standard deviation of one group of channels of the
 * input tensor for the group normalization operator.
 *
 * Given a tensor of shape [W, H, C, N] the mean and standard deviation tensors
 * will have a shape of [G, N] where G = C / group.
 *
 * The input tensor is assumed to be a channels-packed texture tensor with the
 * standard axis mapping. The output tensors are assumed to be contiguous buffer
 * tensors.
 *
 * Algorithm:
 * 1. Each shader invocation corresponds to one group in one batch
 * 2. The local work group cooperatively reduces over all spatial locations (H×W)
 *    and all channels within the group (C/group channels)
 * 3. Uses shared memory for efficient parallel reduction
 * 4. The variance is computed in a second pass over the group as the mean of
 *    the squared deviations from the mean computed in the first pass. Computing
 *    it as E[x^2] - E[x]^2 instead loses all precision when the mean is large
 *    relative to the standard deviation.
 * 5. Main thread (local ID 0) writes the final mean and rstd to buffer
 *
 * Global work group size: {N, 1, 1}
 * N is the number of elements in the tensor buffer; each thread computes one
 * output element.
 *
 * Local work group size:  {1, 1, 64}
 * This allows efficient tree-based reduction in shared memory. Each local
 * group cooperates to compute one output element.
 *
 * Each shader invocation will compute the mean and standard deviation for one
 * channel group in the input, and write out the corresponding result.
 */
void group_norm_reduce_C_packed() {
  const int global_idx = int(gl_GlobalInvocationID.x);
  const int local_idx = int(gl_LocalInvocationID.z);

  // Calculate group dimensions
  const int D = in_sizes.z / group;  // channels per group
  const int HxW = in_sizes.y * in_sizes.x;  // spatial size
  const int group_size = D * HxW;  // total elements per group
  const float count = float(group_size);

  // Convert global index to (group_idx, batch_idx)
  const ivec4 mean_tidx = bufi_to_tidx(global_idx, mean_strides, mean_dim_order);

  // Calculate the range of channels for this group
  const int group_start_channel = mean_tidx.x * D;
  const int group_end_channel = group_start_channel + D;

  // Calculate the range of texels that contain channels from this group
  const int start_texel_idx = group_start_channel / 4;
  const int end_texel_idx = divup4(group_end_channel);
  const int texels_in_group = end_texel_idx - start_texel_idx;

  // Total texels to process across all spatial locations
  const int total_texels = texels_in_group * HxW;

  // Each thread processes a subset of texels
  const int texels_per_thread = (total_texels + LOCAL_WORK_GROUP_SIZE - 1) / LOCAL_WORK_GROUP_SIZE;
  const int start_texel = local_idx * texels_per_thread;
  const int end_texel = min(start_texel + texels_per_thread, total_texels);

  // First pass: sum of the group's values
  float local_sum = 0.0;
  for (int texel_idx = start_texel; texel_idx < end_texel; texel_idx++) {
    bvec4 mask;
    const vec4 texel_val = load_group_texel(
        texel_idx,
        texels_in_group,
        start_texel_idx,
        group_start_channel,
        group_end_channel,
        mean_tidx.y,
        mask);
    // mix() with a bvec selects, so values of other groups never contribute
    local_sum += dot(mix(vec4(0.0), texel_val, mask), vec4(1.0));
  }

  shared_sum[local_idx] = local_sum;
  memoryBarrierShared();
  barrier();

  for (int stride = LOCAL_WORK_GROUP_SIZE / 2; stride > 0; stride /= 2) {
    if (local_idx < stride) {
      shared_sum[local_idx] += shared_sum[local_idx + stride];
    }
    memoryBarrierShared();
    barrier();
  }

  // Every thread needs the mean for the second pass. shared_sum is not written
  // again, so no barrier is needed before shared_sum_sq is reused below.
  const float mean_val = shared_sum[0] / count;

  // Second pass: sum of the squared deviations from the mean
  float local_sum_sq = 0.0;
  for (int texel_idx = start_texel; texel_idx < end_texel; texel_idx++) {
    bvec4 mask;
    const vec4 texel_val = load_group_texel(
        texel_idx,
        texels_in_group,
        start_texel_idx,
        group_start_channel,
        group_end_channel,
        mean_tidx.y,
        mask);
    const vec4 diff = mix(vec4(0.0), texel_val - mean_val, mask);
    local_sum_sq += dot(diff, diff);
  }

  shared_sum_sq[local_idx] = local_sum_sq;
  memoryBarrierShared();
  barrier();

  for (int stride = LOCAL_WORK_GROUP_SIZE / 2; stride > 0; stride /= 2) {
    if (local_idx < stride) {
      shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
    }
    memoryBarrierShared();
    barrier();
  }

  // Main thread writes the result
  if (local_idx == 0 && global_idx < mean_numel) {
    const float variance = shared_sum_sq[0] / count;
    const float rstd_val = 1.0 / sqrt(variance + epsilon);

    // Write to buffer-backed tensors
    t_mean[global_idx] = BUF_T(mean_val);
    t_rstd[global_idx] = BUF_T(rstd_val);
  }
}

void main() {
  group_norm_reduce_C_packed();
}
