/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec3", "in_limits")}
${layout_declare_ubo(B, "vec2", "recip_scales")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int align_corners = 0;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  ivec2 max_in_xy = in_limits.xy - 1;
  vec2 scaled_xy;

  if (align_corners == 1) {
    scaled_xy = pos.xy * recip_scales;
  } else {
    scaled_xy = (pos.xy + 0.5) * recip_scales - 0.5;
  }

  $if MODE == "nearest":
    const ivec2 ipos = clamp(ivec2(round(scaled_xy)), ivec2(0), max_in_xy);
    VEC4_T out_tex = texelFetch(t_in, ivec3(ipos, pos.z), 0);
  $elif MODE == "bilinear":
    vec2 upper_xy = ceil(scaled_xy);
    vec2 lower_xy = floor(scaled_xy);

    // Clamp coordinates to valid input range
    upper_xy = clamp(upper_xy, ivec2(0), max_in_xy);
    lower_xy = clamp(lower_xy, ivec2(0), max_in_xy);

    // Calculate interpolation weights
    vec2 interp_weights = (scaled_xy - lower_xy);

    // Sample the four nearest texels
    VEC4_T sample00 = texelFetch(t_in, ivec3(lower_xy.x, lower_xy.y, pos.z), 0);
    VEC4_T sample10 = texelFetch(t_in, ivec3(upper_xy.x, lower_xy.y, pos.z), 0);
    VEC4_T sample01 = texelFetch(t_in, ivec3(lower_xy.x, upper_xy.y, pos.z), 0);
    VEC4_T sample11 = texelFetch(t_in, ivec3(upper_xy.x, upper_xy.y, pos.z), 0);

    // Perform bilinear interpolation
    VEC4_T out_tex = mix(
      mix(sample00, sample10, interp_weights.x),
      mix(sample01, sample11, interp_weights.x),
      interp_weights.y
    );

  imageStore(t_out, pos, out_tex);
}
