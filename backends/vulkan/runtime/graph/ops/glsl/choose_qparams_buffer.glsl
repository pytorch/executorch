/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define IN_T ${buffer_scalar_type(IN_DTYPE)}

#define ${MODE}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_scale", "float", "buffer")}
${layout_declare_tensor(B, "w", "t_zero_point", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}

$if MODE == "per_tensor":
  layout(push_constant) uniform restrict Block {
    int quant_min;
    int quant_max;
  };
$else:
  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };

${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
${layout_declare_ubo(B, "ivec4", "t_in_strides")}
${layout_declare_ubo(B, "ivec4", "t_scale_sizes")}
${layout_declare_ubo(B, "ivec4", "t_scale_strides")}
${layout_declare_ubo(B, "ivec4", "t_zero_point_sizes")}
${layout_declare_ubo(B, "ivec4", "t_zero_point_strides")}

#include "indexing_utils.h"
#include "choose_qparams.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NWORKERS 64

// Shared memory for reduction - must match local work group size
shared float shared_min[NWORKERS];
shared float shared_max[NWORKERS];

#ifdef per_tensor

void choose_qparams_per_tensor() {
  uint global_id = gl_GlobalInvocationID.x;
  uint local_id = gl_LocalInvocationID.x;
  uint total_threads = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  uint total_elements = uint(t_in_sizes.x * t_in_sizes.y * t_in_sizes.z * t_in_sizes.w);

  // Each thread processes multiple elements with stride
  float thread_min = 1.0/0.0;  // +infinity
  float thread_max = -1.0/0.0; // -infinity
  bool found_valid = false;

  for (uint i = global_id; i < total_elements; i += total_threads) {
    float val = t_in[i];
    if (!isnan(val) && !isinf(val)) {
      if (!found_valid) {
        thread_min = val;
        thread_max = val;
        found_valid = true;
      } else {
        thread_min = min(thread_min, val);
        thread_max = max(thread_max, val);
      }
    }
  }

  // Intra-group reduction using shared memory
  shared_min[local_id] = thread_min;
  shared_max[local_id] = thread_max;
  barrier();

  // Tree reduction within work group
  for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      float other_min = shared_min[local_id + stride];
      float other_max = shared_max[local_id + stride];

      if (!isinf(other_min) && (isinf(shared_min[local_id]) || other_min < shared_min[local_id])) {
        shared_min[local_id] = other_min;
      }
      if (!isinf(other_max) && (isinf(shared_max[local_id]) || other_max > shared_max[local_id])) {
        shared_max[local_id] = other_max;
      }
    }
    barrier();
  }

  // Final result calculation (single workgroup only)
  if (local_id == 0) {
    float global_min = shared_min[0];
    float global_max = shared_max[0];

    float scale_val;
    int zero_point_val;
    calculate_scale_and_zero_point(global_min, global_max, quant_min, quant_max, scale_val, zero_point_val);

    t_scale[0] = scale_val;
    t_zero_point[0] = zero_point_val;
  }
}

#else

void choose_qparams_per_token() {
  uint global_id = gl_GlobalInvocationID.x;
  uint local_id = gl_LocalInvocationID.x;
  uint group_id = gl_WorkGroupID.x;
  uint total_workgroups = gl_NumWorkGroups.x;

  uint total_elements = uint(t_in_sizes.x * t_in_sizes.y * t_in_sizes.z * t_in_sizes.w);
  uint token_size = total_elements / uint(num_tokens);

  // Calculate how many tokens each workgroup should process
  // This handles the case where we have more tokens than workgroups
  uint tokens_per_workgroup = (uint(num_tokens) + total_workgroups - 1) / total_workgroups;

  // Calculate which tokens this workgroup is responsible for
  uint start_token = group_id * tokens_per_workgroup;
  uint end_token = min(start_token + tokens_per_workgroup, uint(num_tokens));

  // Early exit if this workgroup has no tokens to process
  if (start_token >= uint(num_tokens)) {
    return;
  }

  // Process each token assigned to this workgroup
  for (uint token_id = start_token; token_id < end_token; token_id++) {
    // Calculate the start and end indices for this token
    uint token_start = token_id * token_size;
    uint token_end = token_start + token_size;

    // Each thread processes multiple elements within the token with stride
    float thread_min = 1.0/0.0;  // +infinity
    float thread_max = -1.0/0.0; // -infinity
    bool found_valid = false;

    // Process elements within this token only
    for (uint i = token_start + local_id; i < token_end; i += gl_WorkGroupSize.x) {
      float val = t_in[i];
      if (!isnan(val) && !isinf(val)) {
        if (!found_valid) {
          thread_min = val;
          thread_max = val;
          found_valid = true;
        } else {
          thread_min = min(thread_min, val);
          thread_max = max(thread_max, val);
        }
      }
    }

    // Intra-group reduction using shared memory
    shared_min[local_id] = thread_min;
    shared_max[local_id] = thread_max;
    barrier();

    // Tree reduction within work group
    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
      if (local_id < stride) {
        float other_min = shared_min[local_id + stride];
        float other_max = shared_max[local_id + stride];

        if (!isinf(other_min) && (isinf(shared_min[local_id]) || other_min < shared_min[local_id])) {
          shared_min[local_id] = other_min;
        }
        if (!isinf(other_max) && (isinf(shared_max[local_id]) || other_max > shared_max[local_id])) {
          shared_max[local_id] = other_max;
        }
      }
      barrier();
    }

    // Final calculation for this token
    if (local_id == 0) {
      float token_min = shared_min[0];
      float token_max = shared_max[0];

      float scale_val;
      int zero_point_val;
      calculate_scale_and_zero_point(token_min, token_max, quant_min, quant_max, scale_val, zero_point_val);

      t_scale[token_id] = scale_val;
      t_zero_point[token_id] = zero_point_val;
    }

    // Synchronize before processing next token
    barrier();
  }
}

#endif

void main() {
  choose_qparams_${MODE}();
}
