/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

#include "indexing_utils.h"

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
 * Computes a 2D convolution. Each shader invocation calculates the output at
 * a single output location.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_extents.data.xyz))) {
    return;
  }

  // Compute the index of the top-left element of the overlay region. Negative
  // indices indicate that the top-left element is in a region added by padding.
  const ivec2 ipos = pos.xy * params.stride - params.padding;

  // Compute the start and end of the input indices to load. Padding is assumed
  // to be constant 0 padding, so reads from the padding region are skipped.
  const ivec2 start = max(ivec2(0), ipos);
  const ivec2 end = min(ipos + extra_params.overlay_region.xy, ivec2(in_extents.data.xy));
  // Compute the start of the kernel based on how far we are skipping ahead when
  // reading the input. Note that these are "canonical" indices.
  ivec2 kstart = (start - ipos) / params.dilation;
  // During prepacking, the weight tensor was rearranged in order to optimize
  // for data access linearity in this shader. Therefore we need to adjust the
  // canonical coordinates to the corresponding index in the rearranged weight
  // tensor. The x-coordinate is multipled by 4 since each group of 4 channels
  // is folded into the X axis. The y-coordinate is offset based on the z-
  // coordinate because the 2D planes were stacked atop each other vertically.
  kstart.x *= 4;
  kstart.y += pos.z * params.kernel_size.y;

  // Perform the convolution by iterating over the overlay region.
  VEC4_T sum = texelFetch(bias_in, ivec2(pos.z, 0), 0);
  const int ic4 = extra_params.in_group_size / 4;
  for (int z4 = 0; z4 < ic4; ++z4, kstart.x += params.kernel_size.x * 4) {
    for (int y = start.y, ky = kstart.y; y < end.y; y += params.dilation.y, ++ky) {
      for (int x = start.x, kx = kstart.x; x < end.x; x += params.dilation.x, kx += 4) {
        const VEC4_T in_texel = texelFetch(image_in, ivec3(x, y, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        // To explain the calculation below, the contents of in_texel and the
        // group of 4 texels loaded from kernel_in are shown:
        //
        //   in_texel               kernel_in
        //    -x->                   ---x--->
        //   +---+              +----+----+----+----+
        // ^ | w |           ^  | D0 | D1 | D2 | D3 |
        // | +---+           |  +----+----+----+----+
        // | | z |           |  | C0 | C1 | C2 | C3 |
        // z +---+           z  +----+----+----+----+
        // | | y |           |  | B0 | B1 | B2 | B3 |
        // | +---+           |  +----+----+----+----+
        //   | x |              | A0 | A1 | A2 | A3 |
        //   +---+              +----+----+----+----+
        //
        // In the kernel_in graphic, cells sharing the same letter are from
        // the same batch/output channel index, and the number denotes a unique
        // channel index. To calculate the output texel, the following
        // calculation is performed:
        //
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | D0 |   | y | | D1 |   | z | | D2 |   | w | | D3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | C0 |   | y | | C1 |   | z | | C2 |   | w | | C3 |
        //  +---+X+----+ + +---+X+----+ + +---+X+----+ + +---+X+----+
        //  | x | | B0 |   | y | | B1 |   | z | | B2 |   | w | | B3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //  | x | | A0 |   | y | | A1 |   | z | | A2 |   | w | | A3 |
        //  +---+ +----+   +---+ +----+   +---+ +----+   +---+ +----+
        //
        // which is expressed in the following statements.

        sum = fma(in_texel.xxxx, texelFetch(kernel_in, ivec2(kxs.x, ky), 0), sum);
        sum = fma(in_texel.yyyy, texelFetch(kernel_in, ivec2(kxs.y, ky), 0), sum);
        sum = fma(in_texel.zzzz, texelFetch(kernel_in, ivec2(kxs.z, ky), 0), sum);
        sum = fma(in_texel.wwww, texelFetch(kernel_in, ivec2(kxs.w, ky), 0), sum);
      }
    }
  }

  imageStore(image_out, pos, sum);
}
