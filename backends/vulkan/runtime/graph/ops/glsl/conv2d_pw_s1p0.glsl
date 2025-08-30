/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

#define TILE_SIZE_X uint16_t(${TILE_SIZE_X})
#define TILE_SIZE_Y uint16_t(${TILE_SIZE_Y})

#define op(X, A, B) ${OPERATOR}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_limits;
  ivec2 stride;
  ivec2 padding;
  int in_group_size;
  int dummy_padding;
  float out_min;
  float out_max;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "ngroups", "1")}

#extension GL_EXT_control_flow_attributes : require

/*
 * Computes a 2D pointwise convolution of an NxN output tile. Calculating an
 * output tile for pointwise convolution is more efficient because the kernel
 * size is only 1x1, making it easier to re-use loaded texels from t_kernel.
 */
void main() {
  const int out_limits_scaled[2] =
    {(out_limits.x + (TILE_SIZE_X - 1)) / TILE_SIZE_X,
     (out_limits.y + (TILE_SIZE_Y - 1)) / TILE_SIZE_Y};

  const uint16_t div_by_x = uint16_t(gl_GlobalInvocationID.x / out_limits_scaled[0]);
  const uint16_t out_pos_xy[2] = {uint16_t(gl_GlobalInvocationID.x % out_limits_scaled[0]), div_by_x};
  const int out_pos_z = int(gl_GlobalInvocationID.y);

  // If the top left position is out of bounds, then this invocation will have
  // no work to do.
  if (out_pos_xy[1] >= out_limits_scaled[1] || out_pos_z >= out_limits.z) {
    return;
  }

  // Output position for TILE_SIZE = 2
  // +--------+--------+
  // | pos[0] | pos[1] |
  // +--------+--------+
  // | pos[2] | pos[3] |
  // +--------+--------+
  uint16_t pos[TILE_SIZE_X * TILE_SIZE_Y * 2];
  for (uint16_t y = uint16_t(0), i = uint16_t(0); y < TILE_SIZE_Y; ++y) {
    for (uint16_t x = uint16_t(0); x < TILE_SIZE_X; ++x) {
      pos[i * 2] = out_pos_xy[0] * TILE_SIZE_X + x;
      pos[i * 2 + 1] = out_pos_xy[1] * TILE_SIZE_Y + y;
      i++;
    }
  }

  // Final output array where each element is a tensor value.
  // Tuple of consecutive 4 elements represents a single output texel.
  float sum[TILE_SIZE_X * TILE_SIZE_Y * 4];

  // Initialize the output array with the bias value
  for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y * 4; i++) {
    sum[i] = 0;
  }

  int z4 = 0;
  // Since the kernel is 1x1, we only have to loop over the depth dimension.
  for (int z = 0; z < in_group_size; z += 4, ++z4) {
    // During prepacking, the weight tensor has been permuted so that the
    // channel (IC) dim is along the x-axis, and the batch (OC) dim is along
    // the z-axis.
    float kernel_values[4 * 4]; // 4 channels, 4 elements per channel

    // Load kernel values from texels to array
    [[unroll]] for (int i = 0; i < 4; ++i) {
      const vec4 k_tex = texelFetch(t_kernel, ivec2(z + i, out_pos_z), 0);
      kernel_values[i * 4 + 0] = k_tex.x;
      kernel_values[i * 4 + 1] = k_tex.y;
      kernel_values[i * 4 + 2] = k_tex.z;
      kernel_values[i * 4 + 3] = k_tex.w;
    }

    for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
      const vec4 in_tex = texelFetch(t_in, ivec3(pos[i * 2], pos[i * 2 + 1], z4), 0);
      // Load the input texel into an array
      float tex_values[4];
      tex_values[0] = in_tex.x;
      tex_values[1] = in_tex.y;
      tex_values[2] = in_tex.z;
      tex_values[3] = in_tex.w;

      // For 2x2 tile size algorithm works as follows.
      // To explain the calculations below, the contents of one in_tex and the
      // group of 4 texels loaded from t_kernel are shown:
      //
      //   in_tex                 t_kernel
      //    -x->                   ---x--->
      //   +---+              +----+----+----+----+
      // ^ | w |           ^  | D0 | D1 | D2 | D3 |
      // | +---+           |  +----+----+----+----+
      // | | z |           |  | C0 | C1 | C2 | C3 |
      // z +---+           z  +----+----+----+----+
      // | | y |           |  | B0 | B2 | B2 | B3 |
      // | +---+           |  +----+----+----+----+
      //   | x |              | A0 | A1 | A2 | A3 |
      //   +---+              +----+----+----+----+
      //
      // In the t_kernel graphic, cells sharing the same letter are from
      // the same batch/output channel index, and the number denotes a unique
      // channel index. To calculate the output texel, the following
      // calculation is performed:
      //
      //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
      //  | x | | D0 |   | y | | D1 |   | z | | D2 |   | w | | D3 |
      //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
      //  | x | | C0 |   | y | | C1 |   | z | | C2 |   | w | | C3 |
      //  +---+X+----+ + +---+X+----+ + +---+X+----+ + +---+X+----+
      //  | x | | B0 |   | y | | B1 |   | z | | B2 |   | w | | B3 |
      //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
      //  | x | | A0 |   | y | | A1 |   | z | | A2 |   | w | | A3 |
      //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
      //
      //  which is what is expressed in the following calculations. This is done
      //  for each output position.
      for (int j = 0; j < 4; ++j) {
        sum[i * 4 + j] = tex_values[0] * kernel_values[0 + j] + sum[i * 4 + j];
        sum[i * 4 + j] = tex_values[1] * kernel_values[4 + j] + sum[i * 4 + j];
        sum[i * 4 + j] = tex_values[2] * kernel_values[8 + j] + sum[i * 4 + j];
        sum[i * 4 + j] = tex_values[3] * kernel_values[12 + j] + sum[i * 4 + j];
      }
    }
  }

  const vec4 bias = texelFetch(t_bias, ivec2(out_pos_z, 0), 0);

  for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
    const ivec3 pos_l = ivec3(pos[i * 2], pos[i * 2 + 1], out_pos_z);
    if (all(lessThan(pos_l.xy, out_limits.xy))) {
      const vec4 out_sum = vec4(sum[i * 4], sum[i * 4 + 1], sum[i * 4 + 2], sum[i * 4 + 3]);
      imageStore(t_out, pos_l, op(out_sum + bias, out_min, out_max));
    }
  }
}
