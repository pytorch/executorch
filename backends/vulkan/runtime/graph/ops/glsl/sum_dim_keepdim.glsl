/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 2) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

// dim to sum
layout(set = 0, binding = 3) uniform PRECISION restrict DimVal {
  int dim;
};

// size of dim (in the input)
layout(set = 0, binding = 4) uniform PRECISION restrict DimSize {
  int dim_size;
};

layout(set = 0, binding = 5) uniform PRECISION restrict Channel {
  int flattened_channels;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

/*
 * Returns a new tensor with values summed along dimension dim.
 * Output and input have same number of dimensions.
 * summed dimension is of size 1.
 */

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  vec4 out_texel = vec4(0);

  int src_n;
  int src_c;

  // Batch
  if (dim == 0) {
    for (int batch = 0; batch < dim_size; ++batch) {
      src_n = batch;
      src_c = pos.z;
      int src_z = src_n * flattened_channels + src_c;
      out_texel += texelFetch(image_in, ivec3(pos.x, pos.y, src_z), 0);
    }
    imageStore(image_out, pos, out_texel);
  }

  // Channel
  else if (dim == 1) {
    for (int out_index = 0; out_index < 4; ++out_index) {
      for (int channel = 0; channel < dim_size; ++channel) {
        src_n = pos.z;
        src_c = channel;
        int src_z = src_n * flattened_channels + src_c / 4;
        vec4 v = texelFetch(image_in, ivec3(pos.x, pos.y, src_z), 0);
        out_texel[out_index] += v[channel % 4];
      }
    }
    imageStore(image_out, pos, out_texel);
  }

  // Height, Width
  else {
    for (int hw = 0; hw < dim_size; ++hw) {
      vec4 v = (dim == 2)
          ? texelFetch(image_in, ivec3(pos.x, hw, pos.z), 0) // Height
          : texelFetch(image_in, ivec3(hw, pos.y, pos.z), 0); // Width
      out_texel += v;
    }
    imageStore(image_out, pos, out_texel);
  }
}
