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
 * 4. Main thread (local ID 0) writes the final mean and rstd to buffer
 *
 * Global work group size: {N, 1, 1}
 * N is the number of elements in the tensor buffer; each thread computes one
 * output element.
 *
 * Local work group size:  {1, float, 1}
 * float should be a power of 2, recommended 64 or 128 threads. This allows
 * efficient tree-based reduction in shared memory. Each local group will
 * cooperate to compute the output element.
 *
 * Each shader invocation will compute the mean and standard deviation for one
 * channel group in the input, and write out the corresponding result.
 */
void group_norm_reduce_C_packed() {
  const int global_idx = int(gl_GlobalInvocationID.x);
  const int local_idx = int(gl_LocalInvocationID.y);

  // Calculate group dimensions
  const int D = in_sizes.z / group;  // channels per group
  const int HxW = in_sizes.y * in_sizes.x;  // spatial size
  const int group_size = D * HxW;  // total elements per group

  // Convert global index to (group_idx, batch_idx)
  const ivec4 mean_tidx = bufi_to_tidx(global_idx, mean_strides, mean_dim_order);

  // Initialize local sums
  float local_sum = 0.0;
  float local_sum_sq = 0.0;
  int local_count = 0;

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

  // Process assigned texels
  for (int texel_idx = start_texel; texel_idx < end_texel; texel_idx++) {
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
      tex_pos.z += mean_tidx.y * divup4(in_sizes.z);
    }

    // Check bounds and load texel
    if (all(lessThan(tex_pos, in_limits))) {
      const vec4 texel_val = load_texel(t_in, tex_pos);

      // Process all components of the texel that belong to this group
      const int texel_start_channel = global_texel_idx * 4;
      for (int comp = 0; comp < 4; comp++) {
        const int current_channel = texel_start_channel + comp;

        // Check if this component belongs to the current group
        if (current_channel >= group_start_channel && current_channel < group_end_channel) {
          const float val = texel_val[comp];
          local_sum += val;
          local_sum_sq += val * val;
          local_count++;
        }
      }
    }
  }

  // Store local results in shared memory
  shared_sum[local_idx] = local_sum;
  shared_sum_sq[local_idx] = local_sum_sq;

  // Synchronize threads
  memoryBarrierShared();
  barrier();

  // Perform tree-based reduction in shared memory
  for (int stride = LOCAL_WORK_GROUP_SIZE / 2; stride > 0; stride /= 2) {
    if (local_idx < stride) {
      shared_sum[local_idx] += shared_sum[local_idx + stride];
      shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
    }
    memoryBarrierShared();
    barrier();
  }

  // Main thread writes the result
  if (local_idx == 0 && global_idx < mean_numel) {
    const float total_sum = shared_sum[0];
    const float total_sum_sq = shared_sum_sq[0];
    const float count = float(group_size);

    // Calculate mean and reciprocal standard deviation
    const float mean_val = total_sum / count;
    const float variance = (total_sum_sq / count) - (mean_val * mean_val);
    const float rstd_val = 1.0 / sqrt(variance + epsilon);

    // Write to buffer-backed tensors
    t_mean[global_idx] = BUF_T(mean_val);
    t_rstd[global_idx] = BUF_T(rstd_val);
  }
}

void main() {
  group_norm_reduce_C_packed();
}
