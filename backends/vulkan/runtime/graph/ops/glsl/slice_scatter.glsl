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

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_self", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_src", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "selfp")}
${layout_declare_ubo(B, "TextureMetadata", "srcp")}

layout(push_constant) uniform restrict Block {
  // `selected_dim` is in WHCN order; `start`/`end` are already normalized and
  // clamped to [0, self.size(dim)] by add_slice_scatter_node.
  int selected_dim;
  int start;
  int end;
  int step;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "self_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "src_layout", "CONTIG_LAYOUT_INT")}
const int out_packed_dim = get_packed_dim(out_layout);

/*
 * Vulkan implementation of `aten.slice_scatter.default`.
 *
 * out is self with the strided window [start, end) along `dim` replaced by
 * src, which is the gather form
 *
 *   out[..., d, ...] = in_window(d) ? src[..., (d - start) / step, ...]
 *                                   : self[..., d, ...]
 *
 * Writing it as a gather (one read per output element) rather than a copy plus
 * a scatter avoids needing a second pass and keeps every invocation
 * independent, so nothing has to be synchronized between the two sources.
 *
 * The per-component loop matters when `dim` is the packed dim: a single output
 * texel then spans up to four positions along `dim`, which under a step > 1
 * can each independently come from src or from self.
 */
void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx =
      texture_pos_to_tensor4d_idx_simple(outp, out_pos, out_layout);
  VEC4_T out_texel = VEC4_T(0);

  int limit = min(
      4, safe_idx(outp.sizes, out_packed_dim) - out_tidx.data[out_packed_dim]);
  for (int comp = 0; comp < limit; comp++) {
    const int d = out_tidx.data[selected_dim];
    const int rel = d - start;

    if (d >= start && d < end && (rel % step) == 0) {
      TensorIndex4D src_tidx = out_tidx;
      src_tidx.data[selected_dim] = rel / step;
      const TextureElementIndex src_elem =
          tensor4d_idx_to_texture_element_idx_simple(srcp, src_tidx, src_layout);
      out_texel[comp] = texelFetch(t_src, src_elem.pos, 0)[src_elem.comp];
    } else {
      const TextureElementIndex self_elem =
          tensor4d_idx_to_texture_element_idx_simple(
              selfp, out_tidx, self_layout);
      out_texel[comp] = texelFetch(t_self, self_elem.pos, 0)[self_elem.comp];
    }

    out_tidx.data[out_packed_dim]++;
  }

  imageStore(t_out, out_pos, out_texel);
}
