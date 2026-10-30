/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, "buffer")}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
// `t_grid` is always bound as a contiguous (width-packed) buffer of fp scalars
// with logical shape [N, Hout, Wout, 2]. See add_grid_sampler_2d_node which
// asserts this with `is_contiguous_buffer_tensor`.
${layout_declare_tensor(B, "r", "t_grid", DTYPE, "buffer")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// `out_layout` is passed for forward compatibility and is currently asserted
// to be the standard channels-packed layout by `add_grid_sampler_2d_node`.
// All texel math below assumes packed_dim = C (channels-packed), so the four
// fp components of a texel share the same (N, Hout, Wout) and differ only in
// channel. This lets one bilinear interpolation produce all 4 output channels.
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
// padding_mode: 0 = zeros, 1 = border. align_corners: 0 = false, 1 = true.
// Both are specialization constants so the four supported configurations
// share one shader variant and each pipeline still compiles to straight-line
// code with the branches folded away.
${layout_declare_spec_const(C, "int", "padding_mode", "1")}
${layout_declare_spec_const(C, "int", "align_corners", "1")}

/*
 * Vulkan implementation of `aten.grid_sampler_2d.default` for
 *   mode=bilinear, padding_mode in {zeros, border}, align_corners in {0, 1}.
 *
 * RIFE's `WarpModule` uses (border, align_corners=true); the deformable
 * attention in RF-DETR and other DETR derivatives uses (zeros,
 * align_corners=false), which is the torch default.
 *
 * Layout assumptions (validated in add_grid_sampler_2d_node):
 *   - input  : channels-packed texture3d, shape [N, C, Hin, Win]
 *   - grid   : contiguous (width-packed) buffer SSBO of fp scalars,
 *              shape [N, Hout, Wout, 2] in normalized coords [-1, 1]
 *   - output : channels-packed texture3d, shape [N, C, Hout, Wout]
 *
 * For channels-packed texture3d, the texel z extent is N * ceil(C/4),
 * laid out as z = n * num_z_per_n + c_slice. Both input and output share
 * the same N and C, so input z == output z.
 *
 * TextureMetadata layout (vtensor.md): sizes is WHCN order, so
 *   outp.sizes.x = Wout, outp.sizes.y = Hout, outp.sizes.w = N.
 *   outp.limits.z = N * ceil(C/4) (texel slices along z).
 */
/*
 * Read one input texel at integer pixel (c.x, c.y) of channel slice `z`.
 *
 * padding_mode=zeros returns 0 for a corner outside the input, which is what
 * makes the bilinear weights of a partially out-of-range sample sum to less
 * than 1 -- exactly the aten semantics. padding_mode=border clamps instead,
 * which is a no-op when the caller has already clamped the coordinate.
 */
VEC4_T sample_or_zero(const ivec2 c, const ivec2 max_in_xy, const int z) {
  if (padding_mode == 0 &&
      (c.x < 0 || c.y < 0 || c.x > max_in_xy.x || c.y > max_in_xy.y)) {
    return VEC4_T(0);
  }
  const ivec2 cc = clamp(c, ivec2(0), max_in_xy);
  return texelFetch(t_in, ivec3(cc.x, cc.y, z), 0);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  // Derive batch index from texel z. Each batch occupies `num_z_per_n`
  // consecutive z-slices (one per 4-channel slice). Integer division by
  // num_z_per_n picks out the batch.
  const int N = outp.sizes.w;
  const int num_z_per_n = outp.limits.z / N;
  const int n = pos.z / num_z_per_n;

  // Look up the (gx, gy) for this output pixel from the grid SSBO.
  // The grid is a contiguous buffer of [N, Hout, Wout, 2], so the linear
  // index for (n, h, w, comp) is ((n*Hout + h)*Wout + w)*2 + comp. This
  // relies on `inputs_storage` in op_registry.py pinning grid to
  // CONTIGUOUS_BUFFER and the C++ dispatcher re-checking with
  // `is_contiguous_buffer_tensor` — see GridSampler2d.cpp.
  const int Wout = outp.sizes.x;
  const int Hout = outp.sizes.y;
  const int grid_base = ((n * Hout + pos.y) * Wout + pos.x) * 2;
  const float gx_norm = float(t_grid[grid_base + 0]);
  const float gy_norm = float(t_grid[grid_base + 1]);

  // Unnormalize. Input W/H come from inp.sizes (WHCN), not inp.limits
  // (texel space).
  //   align_corners=true : coord = (g + 1) * 0.5 * (size - 1)
  //   align_corners=false: coord = ((g + 1) * size - 1) * 0.5
  // The second form places the normalized range over pixel *edges* rather
  // than pixel centers, so it can legitimately land outside [0, size-1].
  const ivec2 in_size = ivec2(inp.sizes.xy);
  const ivec2 max_in_xy = in_size - 1;
  vec2 g_pixel;
  if (align_corners == 1) {
    g_pixel = (vec2(gx_norm, gy_norm) + 1.0) * 0.5 * vec2(max_in_xy);
  } else {
    g_pixel = ((vec2(gx_norm, gy_norm) + 1.0) * vec2(in_size) - 1.0) * 0.5;
  }

  // padding_mode=border clamps the sample coordinate itself, which also pins
  // the interpolation weights at the edge. padding_mode=zeros must NOT clamp:
  // the weights stay as computed and each out-of-range corner contributes a
  // zero value instead, so clamping here would change the result.
  if (padding_mode == 1) {
    g_pixel = clamp(g_pixel, vec2(0.0), vec2(max_in_xy));
  }

  const ivec2 lower = ivec2(floor(g_pixel));
  const ivec2 upper = lower + ivec2(1);
  const vec2 w = g_pixel - vec2(lower);

  // Fetch the four nearest texels (each carries 4 channels). Because input
  // is channels-packed, pos.z indexes the same channel slice in input as in
  // output, so we can reuse pos.z directly without remapping.
  //
  // For border the coordinate is already clamped, so clamping the corner
  // index is exact. For zeros an out-of-range corner reads as 0.
  VEC4_T s00 = sample_or_zero(ivec2(lower.x, lower.y), max_in_xy, pos.z);
  VEC4_T s10 = sample_or_zero(ivec2(upper.x, lower.y), max_in_xy, pos.z);
  VEC4_T s01 = sample_or_zero(ivec2(lower.x, upper.y), max_in_xy, pos.z);
  VEC4_T s11 = sample_or_zero(ivec2(upper.x, upper.y), max_in_xy, pos.z);

  // Bilinear interpolation. Weights are scalars; mix() acts on all 4 channels.
  VEC4_T out_tex =
      mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);

  imageStore(t_out, pos, out_tex);
}
