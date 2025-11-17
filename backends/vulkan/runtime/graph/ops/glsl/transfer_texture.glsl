/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define UBO_PARAMS ${UBO_PARAMS}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

$if UBO_PARAMS:
  $if OP_NAME == "slice":
    ${layout_declare_ubo(B, "int", "start")}
    ${layout_declare_ubo(B, "int", "step")}

  $if OP_NAME == "select":
    ${layout_declare_ubo(B, "int", "index")}

layout(push_constant) uniform restrict Block {
  int selected_dim;
  $if not UBO_PARAMS:
    $if OP_NAME == "slice":
      int start;
      int step;

    $if OP_NAME == "select":
      int index;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "${OP_NAME}.glslh"

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(outp, out_pos);
  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, outp.sizes[outp.packed_dim] - out_tidx.data[outp.packed_dim]);
  for (int comp = 0; comp < limit; comp++) {
    TensorIndex4D in_tidx = out_tidx_to_in_tidx(out_tidx);

    TextureElementIndex in_elem_pos = tensor4d_idx_to_texture_element_idx_simple(
        inp, in_tidx);

    VEC4_T in_texel = texelFetch(t_in, in_elem_pos.pos, 0);
    out_texel[comp] = in_texel[in_elem_pos.comp];

    out_tidx.data[outp.packed_dim]++;
  }

  imageStore(t_out, out_pos, out_texel);
}
