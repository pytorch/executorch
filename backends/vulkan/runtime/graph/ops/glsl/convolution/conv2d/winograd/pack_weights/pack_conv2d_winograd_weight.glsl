/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, PACKED_STORAGE)}

$if PACKED_STORAGE == "buffer":
  #define OUTPUT_BUFFER

#extension GL_EXT_control_flow_attributes : require

$IN_TILE = OUT_TILE + KERNEL_SIZE - 1
#define IN_TILE ${IN_TILE}
#define KERNEL_SIZE ${KERNEL_SIZE}

${define_required_extensions("buffer", BUF_DTYPE)}
$if PACKED_STORAGE != "buffer":
  ${define_required_extensions(PACKED_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_weight_packed", DTYPE, PACKED_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_src", BUF_DTYPE, "buffer", is_scalar_array=True)}

layout(push_constant) uniform restrict Block {
  // (C_out, C_in)
  ivec2 dims;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_winograd_f${OUT_TILE}x${OUT_TILE}_${KERNEL_SIZE}x${KERNEL_SIZE}.glslh"

// Element (i, j) of G g G^T for the kernel g of output channel co and input
// channel ci, computed in float. Zero for padding channels.
float transformed_weight(const int co, const int ci, const int i, const int j) {
  if (co >= dims.x || ci >= dims.y) {
    return 0.0;
  }
  const int base = (co * dims.y + ci) * (KERNEL_SIZE * KERNEL_SIZE);
  float sum = 0.0;
  [[unroll]] for (int a = 0; a < KERNEL_SIZE; ++a) {
    float row = 0.0;
    [[unroll]] for (int b = 0; b < KERNEL_SIZE; ++b) {
      row += float(t_weight_src[base + a * KERNEL_SIZE + b]) * G[j][b];
    }
    sum += G[i][a] * row;
  }
  return sum;
}

/*
 * Transforms the serialized [C_out, C_in, KERNEL_SIZE, KERNEL_SIZE] weight into
 * the IN_TILE * IN_TILE matrices U_e = G g G^T, e = i * IN_TILE + j, of a
 * Winograd configuration, stacked as batches of the 4OC x 4IC blocked layout
 * that load_packed_weight_x4() reads: texel dk of block (e, k4, n4) holds
 * output channels 4 * n4 to 4 * n4 + 3 of input channel 4 * k4 + dk.
 */
void main() {
  const int n4 = int(gl_GlobalInvocationID.x);
  const int k4 = int(gl_GlobalInvocationID.y);
  const int e = int(gl_GlobalInvocationID.z);

  const int N4 = div_up_4(dims.x);
  const int K4 = div_up_4(dims.y);
  if (n4 >= N4 || k4 >= K4 || e >= IN_TILE * IN_TILE) {
    return;
  }

  const int i = e / IN_TILE;
  const int j = e % IN_TILE;
  [[unroll]] for (int dk = 0; dk < 4; ++dk) {
    const int ci = mul_4(k4) + dk;
    const VEC4_T texel = VEC4_T(
        transformed_weight(mul_4(n4), ci, i, j),
        transformed_weight(mul_4(n4) + 1, ci, i, j),
        transformed_weight(mul_4(n4) + 2, ci, i, j),
        transformed_weight(mul_4(n4) + 3, ci, i, j));
#ifdef OUTPUT_BUFFER
    t_weight_packed[((e * K4 + k4) * N4 + n4) * 4 + dk] = texel;
#else
    imageStore(t_weight_packed, ivec2(mul_4(n4) + dk, e * K4 + k4), texel);
#endif
  }
}
