/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "w", "t_mean", DTYPE, "texture3d")}
${layout_declare_tensor(B, "w", "t_rstd", DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "out_meta")}
${layout_declare_ubo(B, "TextureMetadata", "in_meta")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define MAX_WORKGROUP_SIZE 64

// Shared memory factor increases shared memory allocation by a scale that
// should either be 1 or a power of 2.
//
// Increasing factor allows more data to be stored in shared memory and increase
// thread utilization during reduction. Why? Because when performing reduction,
// the number of active threads becomes half in each iteration. Increasing
// scaling factor increases the thread occupancy and hence utilize the GPU
// better.
#define SHARED_MEMORY_FACTOR 1

#define offset_pos_index(index) ((index) + ((index) >> 3))

shared VEC4_T shared_input[offset_pos_index(MAX_WORKGROUP_SIZE * SHARED_MEMORY_FACTOR)];

ivec3 lpos_to_pos(const ivec3 lpos, const ivec4 axis_map) {
  ivec3 pos;
  pos[axis_map.x] = lpos.x;
  pos[axis_map.y] = lpos.y;
  pos[axis_map.z] = lpos.z;
  return pos;
}

// Reduction of shared memory along the workgroup's x dimension.
void reduce_input(const int width_stride, const int shared_idx_offset) {
  memoryBarrierShared();
  barrier();

  for (int current_stride = 1, index = int(gl_LocalInvocationID.x << 1);
       current_stride < width_stride;
       current_stride *= 2, index <<= 1) {
    if (index < width_stride) {
      const int local_shared_idx = shared_idx_offset + index;
      shared_input[offset_pos_index(local_shared_idx)] +=
          shared_input[offset_pos_index(local_shared_idx + current_stride)];
    }

    memoryBarrierShared();
    barrier();
  }
}

void reduce_non_packed_dim() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const int width = in_meta.sizes.x;
  ivec3 in_pos = lpos_to_pos(lpos, in_meta.axis_map);

  const int width_stride = int(gl_WorkGroupSize.x) * SHARED_MEMORY_FACTOR;

  const int shared_idx_offset =
      width_stride *
      int(gl_WorkGroupSize.y * gl_LocalInvocationID.z +
          gl_LocalInvocationID.y);

  const int shared_idx = shared_idx_offset + int(gl_LocalInvocationID.x);

  VEC4_T mean = VEC4_T(0);
  VEC4_T var = VEC4_T(0);

  for (int width_offset = 0; width_offset < width;
       width_offset += width_stride) {
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      in_pos[in_meta.axis_map.x] =
          width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);

      VEC4_T in_val = VEC4_T(0);
      if (all(lessThan(in_pos, out_meta.limits))) {
        in_val = texelFetch(t_in, in_pos, 0);
      }
      mean += in_val;
    }
  }

  shared_input[offset_pos_index(shared_idx)] = mean;
  reduce_input(width_stride, shared_idx_offset);
  mean = shared_input[offset_pos_index(shared_idx_offset)] / width;

  memoryBarrierShared();
  barrier();

  for (int width_offset = 0; width_offset < width;
       width_offset += width_stride) {
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      in_pos[in_meta.axis_map.x] =
          width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);

      VEC4_T in_val = mean;
      if (all(lessThan(in_pos, out_meta.limits))) {
        in_val = texelFetch(t_in, in_pos, 0);
      }

      const VEC4_T delta = in_val - mean;
      var += delta * delta;
    }
  }

  shared_input[offset_pos_index(shared_idx)] = var;
  reduce_input(width_stride, shared_idx_offset);
  var = shared_input[offset_pos_index(shared_idx_offset)] / width;

  VEC4_T rstd = pow(var + epsilon, VEC4_T(-0.5));
  VEC4_T offset = -rstd * mean;

  const ivec3 in_lpos = lpos;
  const ivec3 in_texel_pos = lpos_to_pos(in_lpos, in_meta.axis_map);
  VEC4_T v = texelFetch(t_in, in_texel_pos, 0);
  VEC4_T weight = texelFetch(t_weight, ivec3(lpos.x, 0, 0), 0).xxxx;
  VEC4_T bias = texelFetch(t_bias, ivec3(lpos.x, 0, 0), 0).xxxx;
  VEC4_T outtex = (v * rstd + offset) * weight + bias;

  if (all(lessThan(lpos, out_meta.limits))) {
    imageStore(t_out, lpos_to_pos(lpos, out_meta.axis_map), outtex);
  }

  if (gl_GlobalInvocationID.x == 0) {
    imageStore(t_mean, lpos_to_pos(lpos, in_meta.axis_map), mean);
    imageStore(t_rstd, lpos_to_pos(lpos, in_meta.axis_map), rstd);
  }
}

void reduce_packed_dim() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const int width = in_meta.sizes.x;
  ivec3 in_pos = lpos_to_pos(lpos, in_meta.axis_map);

  const int width_stride = int(gl_WorkGroupSize.x) * SHARED_MEMORY_FACTOR;

  const int shared_idx_offset =
      width_stride *
      int(gl_WorkGroupSize.y * gl_LocalInvocationID.z +
          gl_LocalInvocationID.y);

  const int shared_idx = shared_idx_offset + int(gl_LocalInvocationID.x);

  const int last_packed_width_index = div_up_4(width) - 1;
  T mean = T(0);
  T var = T(0);
  const int remain = width & 3;

  const int in_pos_x_limit = out_meta.limits[in_meta.axis_map.x];

  VEC4_T accum = VEC4_T(0);
  for (int width_offset = 0; width_offset <= last_packed_width_index;
       width_offset += width_stride) {
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      const int in_pos_x =
          width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);
      in_pos[in_meta.axis_map.x] = in_pos_x;

      VEC4_T in_val = VEC4_T(0);
      if (in_pos_x < in_pos_x_limit) {
        in_val = texelFetch(t_in, in_pos, 0);
      }

      if (in_pos_x == last_packed_width_index && remain != 0) {
        const int remain_inv = 4 - remain;
        in_val.y = mix(in_val.y, T(0), remain_inv > 2);
        in_val.z = mix(in_val.z, T(0), remain_inv > 1);
        in_val.w = mix(in_val.w, T(0), remain_inv > 0);
      }
      accum += in_val;
    }
  }

  shared_input[offset_pos_index(shared_idx)] = accum;
  reduce_input(width_stride, shared_idx_offset);
  VEC4_T val = shared_input[offset_pos_index(shared_idx_offset)];
  mean = (val.x + val.y + val.z + val.w) / width;

  memoryBarrierShared();
  barrier();

  VEC4_T delta2 = VEC4_T(0);

  for (int width_offset = 0; width_offset <= last_packed_width_index;
       width_offset += width_stride) {
    for (int si = 0; si < SHARED_MEMORY_FACTOR; si++) {
      const int in_pos_x =
          width_offset + int(gl_LocalInvocationID.x + si * gl_WorkGroupSize.x);
      in_pos[in_meta.axis_map.x] = in_pos_x;

      VEC4_T in_val = VEC4_T(mean);
      if (in_pos_x < in_pos_x_limit) {
        in_val = texelFetch(t_in, in_pos, 0);
      }

      if (in_pos_x == last_packed_width_index && remain != 0) {
        const int remain_inv = 4 - remain;
        in_val.y = mix(in_val.y, mean.x, remain_inv > 2);
        in_val.z = mix(in_val.z, mean.x, remain_inv > 1);
        in_val.w = mix(in_val.w, mean.x, remain_inv > 0);
      }

      const VEC4_T delta = in_val - mean;
      delta2 += delta * delta;
    }
  }

  shared_input[offset_pos_index(shared_idx)] = delta2;
  reduce_input(width_stride, shared_idx_offset);
  val = shared_input[offset_pos_index(shared_idx_offset)];
  var = (val.x + val.y + val.z + val.w) / width;

  T rstd = pow(var + T(epsilon), T(-0.5));
  T offset = -rstd * mean;

  const ivec3 in_texel_pos = lpos_to_pos(lpos, in_meta.axis_map);
  VEC4_T v = texelFetch(t_in, in_texel_pos, 0);
  VEC4_T weight = texelFetch(t_weight, ivec3(lpos.x, 0, 0), 0);
  VEC4_T bias = texelFetch(t_bias, ivec3(lpos.x, 0, 0), 0);
  VEC4_T outtex = (v * rstd + offset) * weight + bias;

  if (all(lessThan(lpos, out_meta.limits))) {
    imageStore(t_out, lpos_to_pos(lpos, out_meta.axis_map), outtex);
  }

  if (gl_GlobalInvocationID.x == 0) {
    imageStore(
        t_mean,
        lpos_to_pos(lpos, in_meta.axis_map),
        VEC4_T(mean));
    imageStore(
        t_rstd,
        lpos_to_pos(lpos, in_meta.axis_map),
        VEC4_T(rstd));
  }
}

void main() {
  if (in_meta.packed_dim != 0) {
    reduce_non_packed_dim();
  } else {
    reduce_packed_dim();
  }
}
