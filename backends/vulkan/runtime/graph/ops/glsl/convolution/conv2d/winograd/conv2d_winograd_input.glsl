/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}
#define HAS_FLOAT16_ACCUMULATION

$IN_TILE = OUT_TILE + KERNEL_SIZE - 1
$INPUT_GROUPS = IN_TILE // 2 + 1
#define OUT_TILE ${OUT_TILE}
#define IN_TILE ${IN_TILE}
#define PADDING ${(KERNEL_SIZE - 1) // 2}
#define INPUT_GROUPS ${INPUT_GROUPS}

$if ZERO_BORDER:
  #define ZERO_BORDER

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_v", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d", precision="mediump")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(push_constant) uniform restrict Block {
  // The first tile row of this dispatch, and the tile rows t_v holds.
  ivec2 chunk;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_winograd.glslh"
#include "conv2d_winograd_f${OUT_TILE}x${OUT_TILE}_${KERNEL_SIZE}x${KERNEL_SIZE}.glslh"

VEC4_T load_input(const ivec2 p, const int c4, const highp vec3 input_rcp) {
#ifdef ZERO_BORDER
  // The sampler returns zero outside the image, which matches the input
  // exactly.
  return VEC4_T(textureLod(
      t_in, vec3((vec2(p) + 0.5) * input_rcp.xy, (float(c4) + 0.5) * input_rcp.z), 0.0));
#else
  // Padding pixels read the nearest input pixel and are zeroed by a multiply.
  const ivec2 clamped = clamp(p, ivec2(0), in_sizes.xy - 1);
  const float16_t scale = float16_t(all(equal(clamped, p)) ? 1.0 : 0.0);
  return VEC4_T(texelFetch(t_in, ivec3(clamped, c4), 0)) * scale;
#endif
}

// Stores rows i0 and i1 of V for the tile at `origin` in slice c4 of the input,
// at column `tile` of t_v. Pass constant rows, so that only the
// input rows with nonzero coefficients are loaded.
void transform_rows(
    const int i0,
    const int i1,
    const ivec2 origin,
    const int tile,
    const int c4,
    const int C4,
    const highp vec3 input_rcp) {
  VEC4_T x0[IN_TILE];
  VEC4_T x1[IN_TILE];
  [[unroll]] for (int j = 0; j < IN_TILE; ++j) {
    VEC4_T column[IN_TILE];
    [[unroll]] for (int r = 0; r < IN_TILE; ++r) {
      column[r] = load_input(origin + ivec2(j, r), c4, input_rcp);
    }
    x0[j] = bt_transform_element(column, i0);
    x1[j] = bt_transform_element(column, i1);
  }
  [[unroll]] for (int k = 0; k < IN_TILE; ++k) {
    imageStore(
        t_v,
        ivec2(tile, (i0 * IN_TILE + k) * C4 + c4),
        bt_transform_element(x0, k));
    if (i1 != i0) {
      imageStore(
          t_v,
          ivec2(tile, (i1 * IN_TILE + k) * C4 + c4),
          bt_transform_element(x1, k));
    }
  }
}

/*
 * Input transform of Winograd F(m x m, r x r), with m = OUT_TILE and
 * r = KERNEL_SIZE, for a stride 1 convolution with padding PADDING:
 * V = B^T d B for the IN_TILE x IN_TILE input tile d of each output tile in
 * each 4-channel slice. Element e = i * IN_TILE + j of V for tile t of the
 * chunk, in row-major order, is stored at (t, e * C4 + c4) of t_v.
 *
 * INPUT_GROUPS invocations transform each tile. Invocation g stores rows 2g - 1
 * and 2g of V, within the tile: the row of point 0, the rows of a pair +-p, or
 * the row of infinity. Each needs only the input rows with nonzero
 * coefficients. Row groups vary along y, so that the row group of a work group
 * row is uniform.
 */
void main() {
  const int tile_x = int(gl_GlobalInvocationID.x);
  const int row = int(gl_GlobalInvocationID.y / INPUT_GROUPS);
  const int group = int(gl_GlobalInvocationID.y % INPUT_GROUPS);
  const int c4 = int(gl_GlobalInvocationID.z);

  const int C4 = div_up_4(in_sizes.z);
  const int tiles_x = tiles_of(in_sizes.x);
  const int rows = min(chunk.y, tiles_of(in_sizes.y) - chunk.x);
  if (tile_x >= tiles_x || row >= rows || c4 >= C4) {
    return;
  }

#ifdef ZERO_BORDER
  // A highp extent keeps the division at highp: textureSize() of the mediump
  // sampler is mediump, and so would be the reciprocal.
  const highp ivec3 input_extent = textureSize(t_in, 0);
  const highp vec3 input_rcp = vec3(1.0) / vec3(input_extent);
#else
  const highp vec3 input_rcp = vec3(0);
#endif

  const ivec2 origin = ivec2(tile_x, chunk.x + row) * OUT_TILE - PADDING;
  const int tile = row * tiles_x + tile_x;
  // An else for the last branch took fewer instructions on Adreno than a test.
  $for g in range(INPUT_GROUPS):
    $branch = ("else " if g > 0 else "") + ("if (group == %d) " % g if g < INPUT_GROUPS - 1 else "")
    ${branch}{
      transform_rows(${max(2 * g - 1, 0)}, ${min(2 * g, IN_TILE - 1)}, origin, tile, c4, C4, input_rcp);
    }
}
