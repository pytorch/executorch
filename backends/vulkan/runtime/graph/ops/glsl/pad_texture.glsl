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
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}
${layout_declare_ubo(B, "ivec4", "pad_per_dim")}
${layout_declare_ubo(B, "float", "fill_value")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int packed_dim = get_packed_dim(out_layout);

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  // Convert the thread position to output tensor indices in element space.
  // out_tidx.data[packed_dim] is the element index of the first component in
  // this texel; the remaining three dims are scalar element indices.
  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(outp, out_pos);

  // Tail texels may have fewer than 4 valid elements; leave extras as 0.
  const int limit =
      min(4, outp.sizes[packed_dim] - out_tidx.data[packed_dim]);

  VEC4_T out_texel = VEC4_T(0);

  // Process each of the (up to 4) elements in this output texel independently.
  // For each element: subtract pad offsets to obtain the input element index,
  // then copy from the input if in-bounds or write fill_value if in the padding
  // region.
  [[unroll]] for (int comp = 0; comp < limit; comp++) {
    TensorIndex4D in_tidx = out_tidx;
    in_tidx.data[packed_dim] += comp;
    in_tidx.data[0] -= pad_per_dim[0];
    in_tidx.data[1] -= pad_per_dim[1];
    in_tidx.data[2] -= pad_per_dim[2];
    in_tidx.data[3] -= pad_per_dim[3];

    // Signed underflow (output index < pad) produces a negative value that
    // fails the >= 0 check, correctly identifying the padding region.
    if (in_tidx.data[0] >= 0 && in_tidx.data[0] < inp.sizes[0] &&
        in_tidx.data[1] >= 0 && in_tidx.data[1] < inp.sizes[1] &&
        in_tidx.data[2] >= 0 && in_tidx.data[2] < inp.sizes[2] &&
        in_tidx.data[3] >= 0 && in_tidx.data[3] < inp.sizes[3]) {
      TextureElementIndex elem =
          tensor4d_idx_to_texture_element_idx_simple(inp, in_tidx);
      VEC4_T in_texel = texelFetch(t_in, elem.pos, 0);
      out_texel[comp] = T(in_texel[elem.comp]);
    } else {
      out_texel[comp] = T(fill_value);
    }
  }

  imageStore(t_out, out_pos, out_texel);
}
