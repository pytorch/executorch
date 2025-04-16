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

#define VEC4_T ${texel_type(DTYPE)}

#define T ${texel_component_type(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_mean", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_rstd", DTYPE, STORAGE)}

${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE)}

layout(push_constant) uniform PRECISION restrict Block {
  ivec3 out_limits;
  ivec4 sizes;
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);
const lowp int in_packed_dim = unhash_packed_dim(in_layout);

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

#define MAX_WORKGROUP_SIZE 64

// Shared memory factor increases shared memory allocation by a scale that should either be 1 or a power of 2.
//
// Increasing factor allows more data to be stored in shared memory and increase thread utilization during reduction.
// Why? Because when performing reduction, the number of active threads becomes half in each iteration.
// Increasing scaling factor increases the thread occupancy and hence utilize the GPU better.
// eg.
// If local thread size in x dimension is 32, and SHARED_MEMORY_FACTOR is 1, 32 elements will be loaded into shared memory.
// First iteration of reduce will have 16 threads sum up 32 elements.
// Second iteration will have 8 threads sum up 16 elements from previous iteration and so on.
// So thread utilization starts at 50%.
//
// By contrast if local thread size in x dimension is 32, and SHARED_MEMORY_FACTOR is 2, 64 elements will be loaded into shared memory.
// First iteration of reduce will have 32 threads sum up 64 elements.
// Second iteration will have 32 threads sum up 16 elements from previous iteration and so on.
// Thus thread utilization starts at 100%.
#define SHARED_MEMORY_FACTOR 2

#define offset_pos_index(index) ((index) + ((index) >> 2))

shared VEC4_T shared_input[offset_pos_index(MAX_WORKGROUP_SIZE * SHARED_MEMORY_FACTOR)];

// Function to reduce input data in workgroup's x dimension
//
// The implementation resembles reduction as depicted below
// | 10 | 1 | 8 | 1 | 0 | 2 | 3 | 5 | 2 | 3 | 2 | 7 | 0 | 11 | 0 | 2 | current_stride -> 1
//   |   /    |   /   |   /   |   /   |   /   |   /   |   /    |   /
//   |  /     |  /    |  /    |  /    |  /    |  /    |  /     |  /
//   | /      | /     | /     | /     | /     | /     | /      | /
// | 11 | 1 | 9 | 1 | 2 | 2 | 8 | 5 | 5 | 3 | 9 | 7 | 11 | 11 | 2 | 2 | current_stride -> 2
//   |       /        |       /       |       /       |       /
//   |    /           |    /          |    /          |    /
//   | /              | /             | /             | /
// | 20 | 1 | 9 | 1 | 10 | 2 | 8 | 5 |14 | 3 | 9 | 7 |13 | 11 | 2 | 2 | current_stride -> 4
//   |                /               |               /
//   |            /                   |            /
//   |        /                       |         /
//   |    /                           |     /
//   | /                              | /
// | 30 | 1 | 9 | 1 | 10 | 2 | 8 | 5 |27 | 3 | 9 | 7 |13 | 11 | 2 | 2 | current_stride -> 8
//   |                                /
//   |                             /
//   |                         /
//   |                     /
//   |                 /
//   |             /
//   |         /
//   |     /
//   | /
// | 57 | 1 | 9 | 1 | 10 | 2 | 8 | 5 |27 | 3 | 9 | 7 |13 | 11 | 2 | 2 | current_stride = -> 16
//
// Threads access shared index in following pattern
// Thread       | 0 | 1 | 2 | 3 | 4 | 5  | 6  | 7  | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | current_stride -> 1
// Shared Index | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | X | X | X  | X  | X  | X  | X  | X  | index *= 1
//
// Thread       | 0 | 1 | 2 | 3  | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | current_stride -> 2
// Shared Index | 0 | 4 | 8 | 12 | X | X | X | X | X | X | X  | X  | X  | X  | X  | X  | index *= 2
//
// Thread       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | current_stride -> 4
// Shared Index | 0 | 8 | X | X | X | X | X | X | X | X | X  | X  | X  | X  | X  | X  | index *= 4
//
// Thread       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | current_stride -> 8
// Shared Index | 0 | X | X | X | X | X | X | X | X | X | X  | X  | X  | X  | X  | X  | index *= 8

void reduce_input(const int width_stride, const int shared_idx_offset) {
  // wait for all shared memory writes to finish
  memoryBarrierShared();
  barrier();

  // loop log(width_stride) times
  for (int current_stride = 1, index = int(gl_LocalInvocationID.x << 1); current_stride < width_stride; current_stride *= 2, index <<= 1) {
    // if the index at this thread is within the width stride
    if (index < width_stride) {
      const int local_shared_idx = shared_idx_offset + index;
      // add the value at current stride to this thread's value
      shared_input[offset_pos_index(local_shared_idx)] += shared_input[offset_pos_index(local_shared_idx + current_stride)];
    }

    memoryBarrierShared();
    barrier();
  }
}

void reduce_non_packed_dim() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const int width = int(sizes.x);
  ivec3 in_pos = lpos_to_pos(lpos, in_axis_map);

  // width batch read stride
  const int width_stride = int(gl_WorkGroupSize.x) * SHARED_MEMORY_FACTOR;

  // local memory starting offset for this thread
  const int shared_idx_offset = width_stride * int(gl_WorkGroupSize.y * gl_LocalInvocationID.z + gl_LocalInvocationID.y);

  // local memory index for this thread
  const int shared_idx = shared_idx_offset + int(gl_LocalInvocationID.x);

  VEC4_T mean = VEC4_T(0);
  VEC4_T var = VEC4_T(0);

  // Loop over the width in stride increments
  for (int width_offset = 0; width_offset < width; width_offset += width_stride) {
    // Read input in shared memory
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      in_pos[in_axis_map.x] = width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);

      VEC4_T in_val = VEC4_T(0);
      if (all(lessThan(in_pos, out_limits))) {
        in_val = load_texel(t_in, in_pos);
      }
      shared_input[offset_pos_index(shared_idx + si * gl_WorkGroupSize.x)] = in_val;
    }

    reduce_input(width_stride, shared_idx_offset);
    mean += shared_input[offset_pos_index(shared_idx_offset)];
  }

  mean /= width;

  memoryBarrierShared();
  barrier();

  // Loop over the width in stride increments
  for (int width_offset = 0; width_offset < width; width_offset += width_stride) {
    // Read input in shared memory
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      in_pos[in_axis_map.x] = width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);

      VEC4_T in_val = mean;
      if (all(lessThan(in_pos, out_limits))) {
        in_val = load_texel(t_in, in_pos);
      }

      const VEC4_T delta = in_val - mean;
      shared_input[offset_pos_index(shared_idx + si * gl_WorkGroupSize.x)] = delta * delta;
    }

    reduce_input(width_stride, shared_idx_offset);
    var += shared_input[offset_pos_index(shared_idx_offset)];
  }

  var /= width;

  VEC4_T rstd = pow(var + epsilon, VEC4_T(-0.5));
  VEC4_T offset = -rstd * mean;

  VEC4_T v = load_texel(t_in, lpos);
  VEC4_T weight = load_texel(t_weight, ivec3(lpos.x, 0, 0)).xxxx;
  VEC4_T bias = load_texel(t_bias, ivec3(lpos.x, 0, 0)).xxxx;
  VEC4_T outtex = (v * rstd + offset) * weight + bias;

  if (all(lessThan(lpos, out_limits))) {
    write_texel_lpos(t_out, lpos, outtex, out_axis_map);
  }

  if (gl_GlobalInvocationID.x == 0) {
    write_texel(t_mean, lpos, mean);
    write_texel(t_rstd, lpos, rstd);
  }
}

void reduce_packed_dim() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const int width = int(sizes.x);
  ivec3 in_pos = lpos_to_pos(lpos, in_axis_map);

  // width batch read stride
  const int width_stride = int(gl_WorkGroupSize.x) * SHARED_MEMORY_FACTOR;

  // local memory starting offset for this thread
  const int shared_idx_offset = width_stride * int(gl_WorkGroupSize.y * gl_LocalInvocationID.z + gl_LocalInvocationID.y);

  // local memory index for this thread
  const int shared_idx = shared_idx_offset + int(gl_LocalInvocationID.x);

  const int last_packed_width_index = divup4(width) - 1;
  T mean = T(0);
  T var = T(0);
  const int remain = width & 3;

  const int in_pos_x_limit = out_limits[in_axis_map.x];

  // Loop over the width in stride increments
  for (int width_offset = 0; width_offset <= last_packed_width_index; width_offset += width_stride) {
    // Read input in shared memory
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      const int in_pos_x = width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);
      in_pos[in_axis_map.x] = in_pos_x;

      VEC4_T in_val = VEC4_T(0);
      if (in_pos_x < in_pos_x_limit) {
        in_val = load_texel(t_in, in_pos);
      }

      if (in_pos_x == last_packed_width_index && remain != 0) {
        const int remain_inv = 4 - remain;
        in_val.y = mix(in_val.y, T(0), remain_inv > 2);
        in_val.z = mix(in_val.z, T(0), remain_inv > 1);
        in_val.w = mix(in_val.w, T(0), remain_inv > 0);
      }

      shared_input[offset_pos_index(shared_idx + si * gl_WorkGroupSize.x)] = in_val;
    }

    reduce_input(width_stride, shared_idx_offset);
    const VEC4_T val = shared_input[offset_pos_index(shared_idx_offset)];
    mean += val.x + val.y + val.z + val.w;
  }

  mean /= width;

  memoryBarrierShared();
  barrier();

  // Loop over the width in stride increments
  for (int width_offset = 0; width_offset <= last_packed_width_index; width_offset += width_stride) {
    // Read input in shared memory
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      const int in_pos_x = width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);
      in_pos[in_axis_map.x] = in_pos_x;

      VEC4_T in_val = VEC4_T(mean);
      if (in_pos_x < in_pos_x_limit) {
        in_val = load_texel(t_in, in_pos);
      }

      if (in_pos_x == last_packed_width_index && remain != 0) {
        const int remain_inv = 4 - remain;
        in_val.y = mix(in_val.y, mean.x, remain_inv > 2);
        in_val.z = mix(in_val.z, mean.x, remain_inv > 1);
        in_val.w = mix(in_val.w, mean.x, remain_inv > 0);
      }

      const VEC4_T delta = in_val - mean;
      const VEC4_T delta2 = delta * delta;
      shared_input[offset_pos_index(shared_idx + si * gl_WorkGroupSize.x)] = delta2;
    }

    reduce_input(width_stride, shared_idx_offset);
    const VEC4_T val = shared_input[offset_pos_index(shared_idx_offset)];
    var += val.x + val.y + val.z + val.w;
  }

  var /= width;

  T rstd = pow(var + epsilon, T(-0.5));
  T offset = -rstd * mean;

  VEC4_T v = load_texel(t_in, lpos);
  VEC4_T weight = load_texel(t_weight, ivec3(lpos.x, 0, 0));
  VEC4_T bias = load_texel(t_bias, ivec3(lpos.x, 0, 0));
  VEC4_T outtex = (v * rstd + offset) * weight + bias;

  if (all(lessThan(lpos, out_limits))) {
    write_texel_lpos(t_out, lpos, outtex, out_axis_map);
  }

  if (gl_GlobalInvocationID.x == 0) {
    write_texel(t_mean, lpos, VEC4_T(mean));
    write_texel(t_rstd, lpos, VEC4_T(rstd));
  }
}

void main() {
  // if packed dimension width
  if (in_packed_dim != W_DIM) {
    reduce_non_packed_dim();
  } else {
    reduce_packed_dim();
  }
}
