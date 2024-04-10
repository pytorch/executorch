/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler2D kernel_in;
layout(set = 0, binding = 3) uniform PRECISION sampler2D bias_in;

layout(set = 0, binding = 4) uniform PRECISION restrict OutExtents {
  uvec4 data;
}
out_extents;

layout(set = 0, binding = 5) uniform PRECISION restrict InExtents {
  uvec4 data;
}
in_extents;

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  ivec2 kernel_size;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
}
params;

// If fields are separated, SwiftShader cannot identify in_group_size.
layout(set = 0, binding = 7) uniform PRECISION restrict ExtraParams {
  ivec2 overlay_region;
  int in_group_size;
}
extra_params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a 2D transpose convolution. Each shader invocation calculates the
 * output at a single output location. For details, refer to conv2d.glsl which
 * uses a similar approach.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_extents.data.xyz))) {
    return;
  }

  ivec2 ipos = pos.xy + params.padding;

  const ivec2 start = max(
      ivec2(0),
      ivec2(ceil((vec2(ipos) - params.kernel_size + 1) / vec2(params.stride))));
  const ivec2 end =
      min(ivec2(in_extents.data.xy),
          ivec2(floor(vec2(ipos) / vec2(params.stride))) + 1);

  const int ic = extra_params.in_group_size;
  const int kx_stride = ic * (params.stride.x - 1);

  int ky_start = extra_params.overlay_region.y - 1 -
      (ipos.y - params.stride.y * start.y) + pos.z * params.kernel_size.y;
  int kx_start = (extra_params.overlay_region.x - 1 -
                  (ipos.x - params.stride.x * start.x)) * ic;

  ${VEC4_T[DTYPE]} sum = texelFetch(bias_in, ivec2(pos.z, 0), 0);
  for (int y = start.y, ky = ky_start; y < end.y; ++y, ky += params.stride.y) {
    for (int x = start.x, kx = kx_start; x < end.x; ++x, kx += kx_stride) {
      for (int z4 = 0; z4 < ic / 4; ++z4, kx += 4) {
        const ${VEC4_T[DTYPE]} in_texel = texelFetch(image_in, ivec3(x, y, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        sum = fma(in_texel.xxxx, texelFetch(kernel_in, ivec2(kxs.x, ky), 0), sum);
        sum = fma(in_texel.yyyy, texelFetch(kernel_in, ivec2(kxs.y, ky), 0), sum);
        sum = fma(in_texel.zzzz, texelFetch(kernel_in, ivec2(kxs.z, ky), 0), sum);
        sum = fma(in_texel.wwww, texelFetch(kernel_in, ivec2(kxs.w, ky), 0), sum);
      }
    }
  }

  imageStore(image_out, pos, sum);
}
