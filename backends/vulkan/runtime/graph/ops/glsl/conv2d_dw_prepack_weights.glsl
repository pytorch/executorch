/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_type(DTYPE)}
#define SCALAR_T ${texel_component_type(DTYPE)}

#include "indexing_utils.h"

$if DTYPE == "half":
  #extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[2][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  BUF_T buffer_in[];
};

// Corresponds to {1,4,3,9} in the example below.
layout(set = 0, binding = 2) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

// Corresponds to {3,3,1,11} in the example below.
layout(set = 0, binding = 3) uniform PRECISION restrict OriginalSizes {
  ivec4 original_sizes;
};

// Corresponds to {1,12} in the example below.
layout(set = 0, binding = 4) uniform PRECISION restrict PaddedSizes {
  ivec2 padded_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

/*
 * Computes special prepacking for a depthwise convolution. Each shader invocation
 * calculates the input buffer location to read into the desired texel. This
 * packing was originally developed on CPU and that approach is described in the
 * rest of this comment. Refer to the code-level comments, for how we translate
 * it to GPU by reversing the steps.
 *
 * Consider an example weight tensor of size {11,1,3,3}. The following
 * transformations will be applied.
 *
 * 1. Pad the N dim so that it is a multiple of 4. In this case, 1
 * batch of padding is added, producing a tensor of size {12,1,3,3}.
 *      at::pad(x, {0,0,0,0,0,0,0,1}, "constant", 0);
 *
 * 2. Flatten the last two dims by reshaping the tensor:
 *      x.reshape({12,1,9});
 *
 * 3. "Fold" the N dim into the C dim. Split the tensor along the N dim so that
 * each split has 4 channels.
 *      x.reshape({3,4,1,9});
 *
 * 4. Stack the batches on each other vertically by permuting the N and C dims
 * and reshaping the tensor.
 *      x.permute({1,0,2,3}).reshape({4,3,9});
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (any(greaterThanEqual(idx, sizes))) {
    return;
  }

  // As in usual staging shaders, map from GPU texel position to normal CPU
  // buffer indices: (9,3) -> (4,3,9)
  const ivec4 p0 = get_texel_nchw_buffer_ixs(idx, sizes, packed_dim);

  // Re-map the normal CPU buffer indices to special indices, through a series
  // of mappings: reshape is a no-op to the underlying indices, so we only map
  // for pad and permute.
  const int Np = padded_sizes.x;
  const int N = original_sizes.w;
  const int C = original_sizes.z;
  const int H = original_sizes.y;
  const int W = original_sizes.x;

  // Undo step 3 permute: (4,3,1,9) -> (3,4,1,9)
  const ivec4 p1 = swap_adj_dims(p0, 4, (Np / 4), (C * H * W));

  // Undo step 1 pad: (12,1,3,3) -> (11,1,3,3)
  // For values in the padded region, write zero instead of buffer data.
  const ivec4 n = p1 / (C * H * W);
  const ivec4 mask = ivec4(greaterThanEqual(n, ivec4(N)));

  VEC4_T texel = VEC4_T(0);
  if (mask.x == 0) {
    texel.x = SCALAR_T(buffer_in[p1.x]);
  }
  if (mask.y == 0) {
    texel.y = SCALAR_T(buffer_in[p1.y]);
  }
  if (mask.z == 0) {
    texel.z = SCALAR_T(buffer_in[p1.z]);
  }
  if (mask.w == 0) {
    texel.w = SCALAR_T(buffer_in[p1.w]);
  }

  imageStore(image_out, pos.xy, texel);
}
