/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("texture3d", "int8")}

#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

#define NUM_OUTPUTS_PER_WG 1
#define NUM_WORKERS_PER_OUTPUT 64

// Maximum total threads in a work group
#define MAX_THREADS 256

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_scales", DTYPE, "texture3d")}
${layout_declare_tensor(B, "w", "t_zps", "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(push_constant) uniform PushConstants {
  int quant_min;
  int quant_max;
};

// Shared memory for cooperative min/max finding
shared T shared_min[NUM_OUTPUTS_PER_WG][NUM_WORKERS_PER_OUTPUT];
shared T shared_max[NUM_OUTPUTS_PER_WG][NUM_WORKERS_PER_OUTPUT];

const float SMALL_SCALE_THRESHOLD = 6.1e-5;

void calculate_scale_and_zero_point(
    float min_val,
    float max_val,
    int qmin,
    int qmax,
    out float scale,
    out int zero_point) {

  // Extend the [min, max] interval to ensure it contains 0
  min_val = min(min_val, 0.0);
  max_val = max(max_val, 0.0);

  // Calculate scale
  scale = (max_val - min_val) / float(qmax - qmin);

  // Handle special cases for scale
  if (scale == 0.0 || isinf(1.0 / scale)) {
    scale = 0.1;
  }

  // Cut off small scale
  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min_val == 0.0) {
      max_val = SMALL_SCALE_THRESHOLD * float(qmax - qmin);
    } else if (max_val == 0.0) {
      min_val = -SMALL_SCALE_THRESHOLD * float(qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min_val *= amplifier;
      max_val *= amplifier;
    }
  }

  // Zero-point computation
  float zero_point_from_min = float(qmin) - min_val / scale;
  float zero_point_from_max = float(qmax) - max_val / scale;
  float zero_point_from_min_error = abs(float(qmin)) - abs(min_val / scale);
  float zero_point_from_max_error = abs(float(qmax)) - abs(max_val / scale);

  float initial_zero_point = zero_point_from_min_error < zero_point_from_max_error
    ? zero_point_from_min
    : zero_point_from_max;

  // Nudge zero point to be an integer
  int nudged_zero_point;
  if (initial_zero_point < float(qmin)) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > float(qmax)) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = int(round(initial_zero_point));
  }

  zero_point = nudged_zero_point;
}

VEC4_T load_input_x4(const int x4, const int y, const int ntexels_x) {
#ifdef USING_BUFFER
  return t_input[(y * ntexels_x) + x4];
#else
  return texelFetch(t_input, ivec3(x4, y, 0), 0);
#endif
}

void find_min_max_for_row(const int output_y) {
  const int worker_id = int(gl_LocalInvocationID.x);
  const int output_id = int(gl_LocalInvocationID.y);

  if (output_y >= input_sizes.y) {
    return;
  }

  // Input is 2D tensor (height x width), width-packed
  // Each channel corresponds to a row in the tensor
  const int X4 = div_4(input_sizes.x);

  // Initialize thread-local min/max
  T local_min = T(1e30);
  T local_max = T(-1e30);

  // Each thread processes elements along their assigned output_id with stride
  // NUM_WORKERS_PER_OUTPUT
  for (int x4 = worker_id; x4 < X4; x4 += NUM_WORKERS_PER_OUTPUT) {
    VEC4_T in_texel = load_input_x4(x4, output_y, X4);
    for (int i = 0; i < 4; i++) {
      local_min = min(local_min, in_texel[i]);
      local_max = max(local_max, in_texel[i]);
    }
  }

  // Store thread-local results in shared memory
  shared_min[output_id][worker_id] = local_min;
  shared_max[output_id][worker_id] = local_max;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result
  for (int i = NUM_WORKERS_PER_OUTPUT / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_min[output_id][worker_id] = min(
          shared_min[output_id][worker_id],
          shared_min[output_id][worker_id + i]);
      shared_max[output_id][worker_id] = max(
          shared_max[output_id][worker_id],
          shared_max[output_id][worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }
}

void main() {
  const int worker_id = int(gl_LocalInvocationID.x);
  const int output_id = int(gl_LocalInvocationID.y);

  const int output_y4 = int(gl_GlobalInvocationID.y);
  const int output_y = mul_4(output_y4);


  VEC4_T scales_out = VEC4_T(0.0);
  ivec4 zps_out = ivec4(0);

  int limit = min(input_sizes.y - output_y, 4);
  for (int i = 0; i < limit; i++) {
    find_min_max_for_row(output_y + i);

    // Only the first thread in the work group will compute the result
    if (worker_id == 0) {
      float local_min = shared_min[output_id][0];
      float local_max = shared_max[output_id][0];

      float scale;
      int zero_point;

      calculate_scale_and_zero_point(
          local_min, local_max, quant_min, quant_max, scale, zero_point);

      scales_out[i] = T(scale);
      zps_out[i] = zero_point;
    }
  }

  if (worker_id == 0) {
    imageStore(t_scales, ivec3(output_y4, 0, 0), scales_out);
    imageStore(t_zps, ivec3(output_y4, 0, 0), zps_out);
  }

}
