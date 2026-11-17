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
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_flow", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Bilinear warp by a pixel-unit flow field, with border padding:
 *   out[n, c, y, x] = in[n, c, y + flow[n, 1, y, x], x + flow[n, 0, y, x]]
 *
 * in, out: channels-packed texture3d, [N, C, H, W]
 * flow   : channels-packed texture3d, [N, 2, H, W] (one z-slice per batch)
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  const int num_z_per_n = outp.limits.z / outp.sizes.w;
  const int n = pos.z / num_z_per_n;

  const VEC4_T flow_texel = texelFetch(t_flow, ivec3(pos.x, pos.y, n), 0);
  const vec2 c = vec2(pos.xy) + vec2(flow_texel.xy);

  imageStore(
      t_out,
      pos,
      bilinear_border_sample(t_in, c, pos.z, ivec2(inp.sizes.xy) - 1));
}
