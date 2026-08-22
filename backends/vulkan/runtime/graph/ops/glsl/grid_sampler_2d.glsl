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

/*
 * Vulkan implementation of `aten.grid_sampler_2d.default` for the
 * specific configuration used by RIFE's `WarpModule`:
 *   mode=bilinear, padding_mode=border, align_corners=true.
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

  // Unnormalize for align_corners=true:
  //   coord_pixel = (coord_norm + 1) * 0.5 * (size - 1)
  // Input W/H come from inp.sizes (WHCN), not inp.limits (texel space).
  const ivec2 max_in_xy = ivec2(inp.sizes.xy) - 1;
  const float gx_pixel = (gx_norm + 1.0) * 0.5 * float(max_in_xy.x);
  const float gy_pixel = (gy_norm + 1.0) * 0.5 * float(max_in_xy.y);

  // padding_mode=border: clamp coordinates to [0, size-1].
  const float gx = clamp(gx_pixel, 0.0, float(max_in_xy.x));
  const float gy = clamp(gy_pixel, 0.0, float(max_in_xy.y));

  const ivec2 lower = ivec2(floor(vec2(gx, gy)));
  // Clamp ceil to valid range for samples on the border.
  const ivec2 upper = clamp(lower + ivec2(1), ivec2(0), max_in_xy);
  const vec2 w = vec2(gx, gy) - vec2(lower);

  // Fetch the four nearest texels (each carries 4 channels). Because input
  // is channels-packed, pos.z indexes the same channel slice in input as in
  // output, so we can reuse pos.z directly without remapping.
  VEC4_T s00 = texelFetch(t_in, ivec3(lower.x, lower.y, pos.z), 0);
  VEC4_T s10 = texelFetch(t_in, ivec3(upper.x, lower.y, pos.z), 0);
  VEC4_T s01 = texelFetch(t_in, ivec3(lower.x, upper.y, pos.z), 0);
  VEC4_T s11 = texelFetch(t_in, ivec3(upper.x, upper.y, pos.z), 0);

  // Bilinear interpolation. Weights are scalars; mix() acts on all 4 channels.
  VEC4_T out_tex =
      mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);

  imageStore(t_out, pos, out_tex);
}
