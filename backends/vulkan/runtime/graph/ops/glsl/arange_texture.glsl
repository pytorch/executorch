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
${layout_declare_ubo(B, "float", "start")}
${layout_declare_ubo(B, "float", "step")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int packed_dim = get_packed_dim(out_layout);

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);

  // arange output is 1D, so the W dimension holds the element index.
  // Compute the value for each element in the texel along the packed dim.
  VEC4_T outtex = VEC4_T(0);
  int limit = min(
      4, safe_idx(outp.sizes, packed_dim) - out_tidx.data[packed_dim]);
  for (int comp = 0; comp < limit; comp++) {
    int elem_idx = out_tidx.data[0]; // W index is the linear element index
    outtex[comp] = VEC4_T(start + elem_idx * step).x;
    out_tidx.data[packed_dim]++;
  }

  imageStore(t_out, out_pos, outtex);
}
