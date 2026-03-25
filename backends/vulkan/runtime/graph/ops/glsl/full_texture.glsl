/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "float", "fill_value")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int packed_dim = get_packed_dim(out_layout);

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  VEC4_T outtex = VEC4_T(fill_value);

  TensorIndex4D tidx =
      texture_pos_to_tensor4d_idx_simple(outp, pos, out_layout);
  const int packed_dim_size = outp.sizes[packed_dim];
  int packed_idx = tidx.data[packed_dim];

  if (packed_idx + 3 >= packed_dim_size) {
    ivec4 packed_ind = ivec4(packed_idx) + ivec4(0, 1, 2, 3);
    VEC4_T valid_idx = VEC4_T(lessThan(packed_ind, ivec4(packed_dim_size)));
    outtex = outtex * valid_idx;
  }

  imageStore(t_out, pos, outtex);
}
