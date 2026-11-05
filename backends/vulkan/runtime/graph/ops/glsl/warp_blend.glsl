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
${layout_declare_tensor(B, "r", "t_img0", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_img1", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_flow", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_mask", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
// flow + mask share this (internal) spatial size; out/img0/img1 are at the
// original (output) size, which may be larger.
${layout_declare_ubo(B, "TextureMetadata", "flowp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Passed for forward compatibility; the texel math below assumes the standard
// channels-packed layout, asserted in add_warp_blend_node.
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}

/*
 * Fused final upsample + warp + blend for RIFE (Vulkan / final_warp_at_original).
 * Replaces this whole tail of the graph with one dispatch:
 *   flow = interpolate(flow, out_size, bilinear, align_corners=False) * [sx,sy,sx,sy]
 *   mask = interpolate(mask, out_size, bilinear, align_corners=False)
 *   warped0 = grid_sample(img0, base_grid + flow[:, :2] * norm_scale, ...)
 *   warped1 = grid_sample(img1, base_grid + flow[:, 2:] * norm_scale, ...)
 *   merged  = clamp(warped0 * sigmoid(mask) + warped1 * (1 - sigmoid(mask)), 0, 1)
 *
 * Two algebraic simplifications make the grid implicit:
 *   - For align_corners=true grid_sample, the normalize/un-normalize round trip
 *     cancels: the sample pixel coordinate is just (out_px + flow_pixels).
 *   - flow/mask are NOT pre-upsampled on the host; instead this shader samples
 *     them at their internal resolution with the same bilinear/align_corners=
 *     False mapping F.interpolate uses, and rescales the displacement per axis
 *     by sx = out_W/flow_W, sy = out_H/flow_H. (When flow size == out size the
 *     mapping is the identity, so this degrades to a plain same-size warp+blend.)
 *
 * Layout assumptions (validated in add_warp_blend_node):
 *   - img0, img1, out : channels-packed texture3d, [N, 3, Hout, Wout]
 *   - flow            : channels-packed texture3d, [N, 4, Hin, Win]
 *                       (texel.xy = (dx, dy) for img0, texel.zw = for img1,
 *                        in INTERNAL pixel units)
 *   - mask            : channels-packed texture3d, [N, 1, Hin, Win] (raw logits)
 * All tensors have <= 4 channels, so each has one z-slice per batch and the
 * output texel position indexes the matching slice in every input.
 */

// Bilinear sample of channels-packed texture `tex` at pixel coord (cx, cy) on
// texel slice `z`, with padding_mode=border (coords clamped to [0, max_xy]).
#define WARP_SAMPLE(tex, cx, cy, z, max_xy, dst)                       \
  {                                                                    \
    const vec2 c = clamp(vec2(cx, cy), vec2(0.0), vec2(max_xy));       \
    const ivec2 lower = ivec2(floor(c));                              \
    const ivec2 upper = clamp(lower + ivec2(1), ivec2(0), max_xy);    \
    const vec2 w = c - vec2(lower);                                   \
    const VEC4_T s00 = texelFetch(tex, ivec3(lower.x, lower.y, z), 0); \
    const VEC4_T s10 = texelFetch(tex, ivec3(upper.x, lower.y, z), 0); \
    const VEC4_T s01 = texelFetch(tex, ivec3(lower.x, upper.y, z), 0); \
    const VEC4_T s11 = texelFetch(tex, ivec3(upper.x, upper.y, z), 0); \
    dst = mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);           \
  }

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  const ivec2 max_xy = ivec2(outp.sizes.xy) - 1;
  const float px = float(pos.x);
  const float py = float(pos.y);

  // Batch index (flow/mask have one z-slice per batch, same as out here).
  const int num_z_per_n = outp.limits.z / outp.sizes.w;
  const int n = pos.z / num_z_per_n;

  // ---- Bilinear upsample mapping for flow/mask: original -> internal coord.
  // Mirrors F.interpolate(mode="bilinear", align_corners=False):
  //   src = (dst + 0.5) * (in / out) - 0.5, clamped to [0, in-1].
  const vec2 in_xy = vec2(flowp.sizes.xy);
  const vec2 out_xy = vec2(outp.sizes.xy);
  const vec2 scale = in_xy / out_xy; // flow_size / out_size
  vec2 src = (vec2(px, py) + 0.5) * scale - 0.5;
  src = max(src, vec2(0.0));
  const ivec2 in_max = ivec2(in_xy) - 1;
  const ivec2 i0 = min(ivec2(floor(src)), in_max);
  const ivec2 i1 = min(i0 + ivec2(1), in_max);
  const vec2 fr = src - vec2(i0);

  // Bilinear-upsampled flow (4ch) at the internal slice z = n.
  const VEC4_T f00 = texelFetch(t_flow, ivec3(i0.x, i0.y, n), 0);
  const VEC4_T f10 = texelFetch(t_flow, ivec3(i1.x, i0.y, n), 0);
  const VEC4_T f01 = texelFetch(t_flow, ivec3(i0.x, i1.y, n), 0);
  const VEC4_T f11 = texelFetch(t_flow, ivec3(i1.x, i1.y, n), 0);
  const VEC4_T flow_texel = mix(mix(f00, f10, fr.x), mix(f01, f11, fr.x), fr.y);

  // Rescale internal-pixel displacement to original-pixel units (per axis).
  const float sx = 1.0 / scale.x; // out_W / flow_W
  const float sy = 1.0 / scale.y; // out_H / flow_H

  VEC4_T warped0;
  WARP_SAMPLE(
      t_img0,
      px + float(flow_texel.x) * sx,
      py + float(flow_texel.y) * sy,
      pos.z,
      max_xy,
      warped0);
  VEC4_T warped1;
  WARP_SAMPLE(
      t_img1,
      px + float(flow_texel.z) * sx,
      py + float(flow_texel.w) * sy,
      pos.z,
      max_xy,
      warped1);

  // Bilinear-upsampled mask logit (same mapping), then sigmoid in fp32.
  const float m00 = float(texelFetch(t_mask, ivec3(i0.x, i0.y, n), 0).x);
  const float m10 = float(texelFetch(t_mask, ivec3(i1.x, i0.y, n), 0).x);
  const float m01 = float(texelFetch(t_mask, ivec3(i0.x, i1.y, n), 0).x);
  const float m11 = float(texelFetch(t_mask, ivec3(i1.x, i1.y, n), 0).x);
  const float logit =
      mix(mix(m00, m10, fr.x), mix(m01, m11, fr.x), fr.y);
  const float m = 1.0 / (1.0 + exp(-logit));

  VEC4_T merged = warped0 * VEC4_T(m) + warped1 * VEC4_T(1.0 - m);
  merged = clamp(merged, VEC4_T(0.0), VEC4_T(1.0));

  imageStore(t_out, pos, merged);
}
