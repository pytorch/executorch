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

$if DTYPE == "half":
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
  #define VEC4_T f16vec4
$else:
  #define VEC4_T ${texel_type(DTYPE)}


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

  int inputAndOutputWidth = out_limits.x;
  int inputAndOutputHeight = out_limits.y;
  int outputChannel = out_limits.z*4;

  // Divided by 4 because the input channels are packed
  int inputChannel = in_group_size/4;

  int threadHW = int(gl_GlobalInvocationID.x);
  int threadOutChannel = int(gl_GlobalInvocationID.y);

  int xIdx = threadHW % inputAndOutputWidth;
  int yIdx = threadHW / inputAndOutputWidth;

  if (threadHW >= inputAndOutputWidth * inputAndOutputHeight && threadOutChannel >= outputChannel) {
    return;
  }

  VEC4_T outputTexel = VEC4_T(texelFetch(t_bias, ivec2(threadOutChannel, 0), 0));

  VEC4_T inputVec;
  VEC4_T weight1OutputChannelPacked;
  VEC4_T weight2OutputChannelPacked;
  VEC4_T weight3OutputChannelPacked;
  VEC4_T weight4OutputChannelPacked;

  // By unrolling the loop in sets of 4, this significantly reduces the number of branching instructions
  // and enables the compiler to rearrange instructions for more efficient memory retrieval and compute
  for (int inputC = 0; inputC < inputChannel; inputC += 1) {

    inputVec = VEC4_T(texelFetch(t_in, ivec3(xIdx, yIdx, inputC), 0));

    weight1OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 0, threadOutChannel), 0));
    weight2OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 1, threadOutChannel), 0));
    weight3OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 2, threadOutChannel), 0));
    weight4OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 3, threadOutChannel), 0));

    outputTexel[0] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[0], weight2OutputChannelPacked[0], weight3OutputChannelPacked[0], weight4OutputChannelPacked[0]));
    outputTexel[1] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[1], weight2OutputChannelPacked[1], weight3OutputChannelPacked[1], weight4OutputChannelPacked[1]));
    outputTexel[2] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[2], weight2OutputChannelPacked[2], weight3OutputChannelPacked[2], weight4OutputChannelPacked[2]));
    outputTexel[3] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[3], weight2OutputChannelPacked[3], weight3OutputChannelPacked[3], weight4OutputChannelPacked[3]));

    inputC += 1;

    inputVec = VEC4_T(texelFetch(t_in, ivec3(xIdx, yIdx, inputC), 0));

    weight1OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 0, threadOutChannel), 0));
    weight2OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 1, threadOutChannel), 0));
    weight3OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 2, threadOutChannel), 0));
    weight4OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 3, threadOutChannel), 0));

    outputTexel[0] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[0], weight2OutputChannelPacked[0], weight3OutputChannelPacked[0], weight4OutputChannelPacked[0]));
    outputTexel[1] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[1], weight2OutputChannelPacked[1], weight3OutputChannelPacked[1], weight4OutputChannelPacked[1]));
    outputTexel[2] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[2], weight2OutputChannelPacked[2], weight3OutputChannelPacked[2], weight4OutputChannelPacked[2]));
    outputTexel[3] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[3], weight2OutputChannelPacked[3], weight3OutputChannelPacked[3], weight4OutputChannelPacked[3]));

    inputC += 1;

    inputVec = VEC4_T(texelFetch(t_in, ivec3(xIdx, yIdx, inputC), 0));

    weight1OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 0, threadOutChannel), 0));
    weight2OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 1, threadOutChannel), 0));
    weight3OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 2, threadOutChannel), 0));
    weight4OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 3, threadOutChannel), 0));

    outputTexel[0] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[0], weight2OutputChannelPacked[0], weight3OutputChannelPacked[0], weight4OutputChannelPacked[0]));
    outputTexel[1] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[1], weight2OutputChannelPacked[1], weight3OutputChannelPacked[1], weight4OutputChannelPacked[1]));
    outputTexel[2] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[2], weight2OutputChannelPacked[2], weight3OutputChannelPacked[2], weight4OutputChannelPacked[2]));
    outputTexel[3] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[3], weight2OutputChannelPacked[3], weight3OutputChannelPacked[3], weight4OutputChannelPacked[3]));

    inputC += 1;

    inputVec = VEC4_T(texelFetch(t_in, ivec3(xIdx, yIdx, inputC), 0));

    weight1OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 0, threadOutChannel), 0));
    weight2OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 1, threadOutChannel), 0));
    weight3OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 2, threadOutChannel), 0));
    weight4OutputChannelPacked = VEC4_T(texelFetch(t_kernel, ivec2(inputC * 4 + 3, threadOutChannel), 0));

    outputTexel[0] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[0], weight2OutputChannelPacked[0], weight3OutputChannelPacked[0], weight4OutputChannelPacked[0]));
    outputTexel[1] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[1], weight2OutputChannelPacked[1], weight3OutputChannelPacked[1], weight4OutputChannelPacked[1]));
    outputTexel[2] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[2], weight2OutputChannelPacked[2], weight3OutputChannelPacked[2], weight4OutputChannelPacked[2]));
    outputTexel[3] += dot(inputVec, VEC4_T(weight1OutputChannelPacked[3], weight2OutputChannelPacked[3], weight3OutputChannelPacked[3], weight4OutputChannelPacked[3]));
  }

  imageStore(t_out, ivec3(xIdx, yIdx, threadOutChannel), op(vec4(outputTexel), out_min, out_max));
}
