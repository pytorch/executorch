/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}
${define_required_extensions(INDEX_STORAGE, "int")}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_self", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_index", "int", INDEX_STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
$if INDEX_STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "index")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "index")}

layout(push_constant) uniform restrict Block {
  ivec2 index_params;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

// Implements aten.index.Tensor with exactly one index tensor. Each output
// element is:
//   output[...] = self[index[...]]
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "index_layout", "CONTIG_LAYOUT_INT")}

int load_index(const TensorIndex out_tidx) {
$if INDEX_STORAGE == "buffer":
  uint index_bufi = 0;
  for (int d = 0; d < index_params.y; ++d) {
    index_bufi +=
        stride_at(index, d) * idx_at(out_tidx, index_params.x + d);
  }
  return t_index[index_bufi];
$else:
  TensorIndex4D index_tidx = zero_tensor4d_idx();
  index_tidx.data.x = int(idx_at(out_tidx, index_params.x));
  if (index_params.y > 1) {
    index_tidx.data.y = int(idx_at(out_tidx, index_params.x + 1));
  }
  if (index_params.y > 2) {
    index_tidx.data.z = int(idx_at(out_tidx, index_params.x + 2));
  }
  if (index_params.y > 3) {
    index_tidx.data.w = int(idx_at(out_tidx, index_params.x + 3));
  }
  const TextureElementIndex index_elem =
      tensor4d_idx_to_texture_element_idx_simple(
          index, index_tidx, index_layout);
  return texelFetch(t_index, index_elem.pos, 0)[index_elem.comp];
}

uint self_idx_at(
    const TensorIndex out_tidx,
    const int self_axis,
    const uint index_value) {
  if (self_axis == index_params.x) {
    return index_value;
  }
  const int out_axis = self_axis < index_params.x
      ? self_axis
      : self_axis + index_params.y - 1;
  return idx_at(out_tidx, out_axis);
}

void main() {
  const uint out_bufi = linear_idx_from_gid();
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  // Convert output buffer index to tensor index
  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

  const int idx = load_index(out_tidx);

  TensorIndex self_tidx;
  initialize(self_tidx);
  const int self_rank = int_ndim(inp);
  if (self_rank > 0) self_tidx.data[0].x = self_idx_at(out_tidx, 0, uint(idx));
  if (self_rank > 1) self_tidx.data[0].y = self_idx_at(out_tidx, 1, uint(idx));
  if (self_rank > 2) self_tidx.data[0].z = self_idx_at(out_tidx, 2, uint(idx));
  if (self_rank > 3) self_tidx.data[0].w = self_idx_at(out_tidx, 3, uint(idx));
  if (self_rank > 4) self_tidx.data[1].x = self_idx_at(out_tidx, 4, uint(idx));
  if (self_rank > 5) self_tidx.data[1].y = self_idx_at(out_tidx, 5, uint(idx));
  if (self_rank > 6) self_tidx.data[1].z = self_idx_at(out_tidx, 6, uint(idx));
  if (self_rank > 7) self_tidx.data[1].w = self_idx_at(out_tidx, 7, uint(idx));
  const uint self_bufi = tensor_idx_to_linear_idx(inp, self_tidx);

  t_out[out_bufi] = t_self[self_bufi];
}
