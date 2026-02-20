/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int split_dim = 0;
layout(constant_id = 4) const int split_idx = 0;
layout(constant_id = 5) const int split_offset = 0;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(outp, out_pos);

  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, outp.sizes[outp.packed_dim] - out_tidx.data[outp.packed_dim]);

  TensorIndex4D input_tidx = out_tidx;
  input_tidx.data[split_dim] += split_offset;

  for (int comp = 0; comp < limit; comp++) {
    TextureElementIndex input_elem_pos = tensor4d_idx_to_texture_element_idx_simple(
        inp, input_tidx);

    VEC4_T input_texel = texelFetch(t_input, input_elem_pos.pos, 0);
    out_texel[comp] = input_texel[input_elem_pos.comp];

    input_tidx.data[outp.packed_dim]++;
  }

  imageStore(t_output, out_pos, out_texel);
}
