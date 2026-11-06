/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Binary comparison ops write a bool/uint8 output dtype, which differs from
// the input dtype. IS_COMPARISON_OP is set explicitly per shader variant in the
// .yaml.

#version 450 core

$PROMOTED_DTYPE = get_higher_precision_dtype(DTYPE, SCALAR_VALUE_TYPE)
$OUT_DTYPE = "uint8" if IS_COMPARISON_OP else DTYPE

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions(STORAGE, PROMOTED_DTYPE)}
${define_explicit_type_extensions(SCALAR_VALUE_TYPE)}
${define_explicit_type_extensions(PROMOTED_DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions(STORAGE, "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define SCALAR_T ${buffer_scalar_type(SCALAR_VALUE_TYPE)}
#define COMPUTE_VEC4_T ${texel_type(PROMOTED_DTYPE)}
#define COMPUTE_T ${texel_component_type(PROMOTED_DTYPE)}
#define OUT_VEC4_T ${texel_load_type(OUT_DTYPE, STORAGE)}
#define OUT_COMPUTE_VEC4_T ${texel_type(OUT_DTYPE)}

#define op(X, Y) ${OPERATOR}

$if not IS_COMPARISON_OP and DTYPE in ("float", "half"):
  #define ROUND_HALF_OUTPUT

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"
#include "convert.glslh"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
  ${layout_declare_ubo(B, "BufferMetadata", "inp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}
  ${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  SCALAR_T scalar_value;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 3) const bool input_is_half = false;
layout(constant_id = 4) const int out_layout = CONTIG_LAYOUT_INT;
const int out_packed_dim = get_packed_dim(out_layout);

#include "dispatch.glslh"

$if not IS_COMPARISON_OP:
  #include "binary_op_defs.glslh"

void main() {
#ifdef USING_BUFFER
  const uint out_idx = linear_idx_from_gid();
  if (out_idx >= div_4(numel(outp))) {
    return;
  }
  const VEC4_T in_texel = t_in[out_idx];
#else
  const ivec3 out_idx = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(out_idx, outp)) {
    return;
  }
  const VEC4_T in_texel = texelFetch(t_in, out_idx, 0);
#endif

  OUT_COMPUTE_VEC4_T out_texel = OUT_COMPUTE_VEC4_T(
      op(COMPUTE_VEC4_T(in_texel), COMPUTE_VEC4_T(scalar_value)));

#ifdef ROUND_HALF_OUTPUT
  if (input_is_half) {
    out_texel = round_to_half_rte(out_texel);
  }
#endif

  out_texel = mask_padding_lanes(out_texel, out_idx, outp, out_packed_dim);

#ifdef USING_BUFFER
  t_out[out_idx] = OUT_VEC4_T(out_texel);
#else
  imageStore(t_out, out_idx, OUT_VEC4_T(out_texel));
#endif
}
