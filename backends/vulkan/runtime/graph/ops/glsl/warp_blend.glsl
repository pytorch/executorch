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
#include "bilinear_border_sample.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_img0", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_img1", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_flow", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_mask", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
// flow and mask share this spatial size, which may be smaller than out's.
${layout_declare_ubo(B, "TextureMetadata", "flowp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Blends two images, each warped by its own pixel-unit flow field:
 *   flow = interpolate(flow, out_size, bilinear, align_corners=False)
 *          * [sx, sy, sx, sy]
 *   mask = interpolate(mask, out_size, bilinear, align_corners=False)
 *   out  = clamp(warp(img0, flow[:, 0:2]) * sigmoid(mask)
 *                + warp(img1, flow[:, 2:4]) * (1 - sigmoid(mask)), 0, 1)
 * where warp is a bilinear sample at (x, y) + flow with border padding, and
 * sx, sy = out size / flow size. The upsample is done in place by sampling flow
 * and mask at their own size; when the sizes match it is the identity.
 *
 * All tensors are channels-packed texture3d. flow and mask have one z-slice
 * per batch.
 *   img0, img1, out: [N, C, H, W]
 *   flow           : [N, 4, h, w]
 *   mask           : [N, 1, h, w]
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  const int num_z_per_n = outp.limits.z / outp.sizes.w;
  const int n = pos.z / num_z_per_n;

  // Bilinear upsample coordinates, as F.interpolate(align_corners=False):
  //   src = (dst + 0.5) * (in / out) - 0.5, clamped to [0, in - 1].
  const vec2 in_xy = vec2(flowp.sizes.xy);
  const vec2 scale = in_xy / vec2(outp.sizes.xy);
  const vec2 src = max((vec2(pos.xy) + 0.5) * scale - 0.5, vec2(0.0));
  const ivec2 in_max = ivec2(in_xy) - 1;
  const ivec2 i0 = min(ivec2(floor(src)), in_max);
  const ivec2 i1 = min(i0 + ivec2(1), in_max);
  const vec2 fr = src - vec2(i0);

  const VEC4_T f00 = texelFetch(t_flow, ivec3(i0.x, i0.y, n), 0);
  const VEC4_T f10 = texelFetch(t_flow, ivec3(i1.x, i0.y, n), 0);
  const VEC4_T f01 = texelFetch(t_flow, ivec3(i0.x, i1.y, n), 0);
  const VEC4_T f11 = texelFetch(t_flow, ivec3(i1.x, i1.y, n), 0);
  const vec4 flow_texel =
      vec4(mix(mix(f00, f10, fr.x), mix(f01, f11, fr.x), fr.y));

  // Displacements are in flow pixels; rescale them to output pixels.
  const vec2 disp_scale = 1.0 / scale;
  const vec2 px = vec2(pos.xy);
  const ivec2 max_xy = ivec2(outp.sizes.xy) - 1;
  const VEC4_T warped0 = bilinear_border_sample(
      t_img0, px + flow_texel.xy * disp_scale, pos.z, max_xy);
  const VEC4_T warped1 = bilinear_border_sample(
      t_img1, px + flow_texel.zw * disp_scale, pos.z, max_xy);

  const float m00 = float(texelFetch(t_mask, ivec3(i0.x, i0.y, n), 0).x);
  const float m10 = float(texelFetch(t_mask, ivec3(i1.x, i0.y, n), 0).x);
  const float m01 = float(texelFetch(t_mask, ivec3(i0.x, i1.y, n), 0).x);
  const float m11 = float(texelFetch(t_mask, ivec3(i1.x, i1.y, n), 0).x);
  const float logit = mix(mix(m00, m10, fr.x), mix(m01, m11, fr.x), fr.y);
  const float m = 1.0 / (1.0 + exp(-logit));

  const VEC4_T merged = warped0 * VEC4_T(m) + warped1 * VEC4_T(1.0 - m);
  imageStore(t_out, pos, clamp(merged, VEC4_T(0.0), VEC4_T(1.0)));
}
