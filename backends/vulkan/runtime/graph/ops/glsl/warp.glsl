/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_flow", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Passed for forward compatibility; the texel math below assumes the standard
// channels-packed layout, asserted in add_warp_node.
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}

/*
 * Single bilinear/border warp WITHOUT an explicitly constructed grid. Replaces
 * RIFE's WarpModule (`grid = base_grid + flow * norm_scale; grid_sample(...)`)
 * for any channel count.
 *
 * For align_corners=true the normalize/un-normalize round trip cancels, so the
 * sample pixel coordinate is simply (out_px + dx, out_py + dy). `flow` is
 * consumed directly in pixel units — no grid tensor and no base_grid buffer.
 *
 * Layout assumptions (validated in add_warp_node):
 *   - in, out : channels-packed texture3d, [N, C, H, W] (C arbitrary)
 *   - flow    : channels-packed texture3d, [N, 2, H, W] (texel.x = dx, .y = dy)
 *
 * For channels-packed texture3d the z extent is N * ceil(C/4). in and out share
 * N and C so input z == output z. flow has 2 channels -> one z-slice per batch,
 * so its slice index is the batch index n = pos.z / num_z_per_n.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  const int N = outp.sizes.w;
  const int num_z_per_n = outp.limits.z / N;
  const int n = pos.z / num_z_per_n;

  const ivec2 max_xy = ivec2(inp.sizes.xy) - 1;

  // flow texel for this output pixel on batch n. padding_mode=border: clamp the
  // sample coordinate to [0, size-1].
  const VEC4_T flow_texel = texelFetch(t_flow, ivec3(pos.x, pos.y, n), 0);
  const float cx =
      clamp(float(pos.x) + float(flow_texel.x), 0.0, float(max_xy.x));
  const float cy =
      clamp(float(pos.y) + float(flow_texel.y), 0.0, float(max_xy.y));

  const ivec2 lower = ivec2(floor(vec2(cx, cy)));
  const ivec2 upper = clamp(lower + ivec2(1), ivec2(0), max_xy);
  const vec2 w = vec2(cx, cy) - vec2(lower);

  // Same channel slice (pos.z) in input as output -> reuse pos.z directly.
  const VEC4_T s00 = texelFetch(t_in, ivec3(lower.x, lower.y, pos.z), 0);
  const VEC4_T s10 = texelFetch(t_in, ivec3(upper.x, lower.y, pos.z), 0);
  const VEC4_T s01 = texelFetch(t_in, ivec3(lower.x, upper.y, pos.z), 0);
  const VEC4_T s11 = texelFetch(t_in, ivec3(upper.x, upper.y, pos.z), 0);

  const VEC4_T out_tex = mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);

  imageStore(t_out, pos, out_tex);
}
