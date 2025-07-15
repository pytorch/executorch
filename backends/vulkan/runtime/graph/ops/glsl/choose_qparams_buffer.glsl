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
    float eps;
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

/*
 * QUANTIZATION PARAMETER COMPUTATION SHADER (BUFFER STORAGE)
 *
 * This shader computes quantization parameters (scale and zero_point) for converting
 * floating-point tensors to n-bit integer representations while preserving the
 * original data range as much as possible.
 *
 * ALGORITHM:
 * 1. Find global min/max values across tensor elements using parallel reduction
 * 2. Use tree reduction with shared memory for efficient min/max computation
 * 3. Calculate scale = (max - min) / (quant_max - quant_min)
 * 4. Calculate zero_point to map floating-point zero to integer value
 *
 * WORKGROUP CONFIGURATION:
 * - Per-Tensor Mode:
 *   - Global WG Size: {1, 1, 1} (single workgroup processes entire tensor)
 *   - Local WG Size: {64, 1, 1} (matches NWORKERS for shared memory)
 * - Per-Token Mode:
 *   - Global WG Size: {num_tokens, 1, 1} (one workgroup per token)
 *   - Local WG Size: {64, 1, 1} (matches NWORKERS for shared memory)
 *
 * SUPPORTED CONFIGURATIONS:
 * - Buffer Storage: Uses simple linear indexing through buffer elements
 * - No axis mapping or packing considerations - processes elements sequentially
 * - Works with any tensor layout since it accesses buffer data linearly
 *
 * TREE REDUCTION VISUALIZATION FOR MIN/MAX FINDING:
 * For 8 threads processing elements [10, 1, 8, 1, 0, 2, 3, 5]:
 *
 * Initial shared_min/shared_max arrays populated by each thread:
 * shared_min:  | 10 | 1 | 8 | 1 | 0 | 2 | 3 | 5 |
 * shared_max:  | 10 | 1 | 8 | 1 | 0 | 2 | 3 | 5 |
 * Thread:      |  0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
 *
 * Stride 1 (compare pairs, keep min/max):
 * shared_min:  |  1 |   | 1 |   | 0 |   | 3 |   |  (min(10,1), min(8,1), min(0,2), min(3,5))
 * shared_max:  | 10 |   | 8 |   | 2 |   | 5 |   |  (max(10,1), max(8,1), max(0,2), max(3,5))
 * Active:      |  0 |   | 2 |   | 4 |   | 6 |   |
 *
 * Stride 2 (compare pairs, keep min/max):
 * shared_min:  |  0 |   |   |   | 0 |   |   |   |  (min(1,1), min(0,3))
 * shared_max:  | 10 |   |   |   | 5 |   |   |   |  (max(10,8), max(2,5))
 * Active:      |  0 |   |   |   | 4 |   |   |   |
 *
 * Stride 4 (final comparison):
 * shared_min:  |  0 |   |   |   |   |   |   |   |  (min(0,0) = 0)
 * shared_max:  | 10 |   |   |   |   |   |   |   |  (max(10,5) = 10)
 * Active:      |  0 |   |   |   |   |   |   |   |
 *
 * Final result: global_min = 0, global_max = 10 (stored in shared_min[0], shared_max[0])
 *
 * PER-TENSOR QUANTIZATION:
 * - Single workgroup processes entire tensor with strided access
 * - Each thread processes elements [thread_id, thread_id + 64, thread_id + 128, ...]
 * - Tree reduction combines all thread results into global min/max
 * - Output: Single scale and zero_point values
 *
 * PER-TOKEN QUANTIZATION:
 * - Multiple workgroups, each processing one token
 * - Token = all elements except last dimension (e.g., for [B,S,H]: B*S tokens of H elements)
 * - Each workgroup finds min/max within its assigned token
 * - Output: Array of scale and zero_point values (one per token)
 */

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
    calculate_scale_and_zero_point(global_min, global_max, quant_min, quant_max, eps, scale_val, zero_point_val);

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
      calculate_scale_and_zero_point(token_min, token_max, quant_min, quant_max, 1e-5, scale_val, zero_point_val);

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
