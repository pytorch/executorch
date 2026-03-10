/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions(STORAGE, "bool")}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${buffer_scalar_type(DTYPE)}
#define COND_T ${buffer_scalar_type("bool")}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_condition", "bool", STORAGE)}
${layout_declare_tensor(B, "r", "t_self", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_other", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
  ${layout_declare_ubo(B, "BufferMetadata", "condp")}
  ${layout_declare_ubo(B, "BufferMetadata", "selfp")}
  ${layout_declare_ubo(B, "BufferMetadata", "otherp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}
  ${layout_declare_ubo(B, "TextureMetadata", "condp")}
  ${layout_declare_ubo(B, "TextureMetadata", "selfp")}
  ${layout_declare_ubo(B, "TextureMetadata", "otherp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef USING_BUFFER

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

  TensorIndex cond_tidx = out_tidx;
  clamp_tensor_idx(condp, cond_tidx);

  TensorIndex self_tidx = out_tidx;
  clamp_tensor_idx(selfp, self_tidx);

  TensorIndex other_tidx = out_tidx;
  clamp_tensor_idx(otherp, other_tidx);

  const uint cond_bufi = tensor_idx_to_linear_idx(condp, cond_tidx);
  const uint self_bufi = tensor_idx_to_linear_idx(selfp, self_tidx);
  const uint other_bufi = tensor_idx_to_linear_idx(otherp, other_tidx);

  COND_T cond = t_condition[cond_bufi];
  T v_self = t_self[self_bufi];
  T v_other = t_other[other_bufi];

  if (cond > 0) {
    t_out[out_bufi] = v_self;
  } else {
    t_out[out_bufi] = v_other;
  }
}

#else // USING_TEXTURE

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(out_pos, outp)) {
    return;
  }

  TensorIndex4D out_tidx = texture_pos_to_tensor4d_idx_simple(outp, out_pos);

  VEC4_T outtex = VEC4_T(0);

  int limit = min(
      4, outp.sizes[outp.packed_dim] - out_tidx.data[outp.packed_dim]);
  for (int comp = 0; comp < limit; comp++) {
    TensorIndex4D cond_tidx;
    cond_tidx.data = min(out_tidx.data, condp.sizes - 1);
    TextureElementIndex cond_elem =
        tensor4d_idx_to_texture_element_idx_simple(condp, cond_tidx);
    uint cond_val = texelFetch(t_condition, cond_elem.pos, 0)[cond_elem.comp];

    TensorIndex4D self_tidx;
    self_tidx.data = min(out_tidx.data, selfp.sizes - 1);
    TextureElementIndex self_elem =
        tensor4d_idx_to_texture_element_idx_simple(selfp, self_tidx);
    VEC4_T self_texel = texelFetch(t_self, self_elem.pos, 0);

    TensorIndex4D other_tidx;
    other_tidx.data = min(out_tidx.data, otherp.sizes - 1);
    TextureElementIndex other_elem =
        tensor4d_idx_to_texture_element_idx_simple(otherp, other_tidx);
    VEC4_T other_texel = texelFetch(t_other, other_elem.pos, 0);

    if (cond_val > 0) {
      outtex[comp] = self_texel[self_elem.comp];
    } else {
      outtex[comp] = other_texel[other_elem.comp];
    }

    out_tidx.data[outp.packed_dim]++;
  }

  imageStore(t_out, out_pos, outtex);
}
#endif
