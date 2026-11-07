/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The op is selected by the op_type specialization constant. Comparison ops
// write a bool/uint8 output dtype, which differs from the input dtype, so they
// get their own variants with IS_COMPARISON_OP set.

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

$if IS_COMPARISON_OP:
  #define BINARY_SCALAR_IS_COMPARISON
$elif DTYPE in ("float", "half"):
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
layout(constant_id = 5) const int op_type = 0;

#include "dispatch.glslh"

$if not IS_COMPARISON_OP:
  #include "binary_op_defs.glslh"

// Values of op_type. Must match BinaryOpType in BinaryOp.h.
#define BINARY_OP_MUL 2
#define BINARY_OP_POW 5
#define BINARY_OP_EQ 8
#define BINARY_OP_LT 9
#define BINARY_OP_LE 10
#define BINARY_OP_GT 11
#define BINARY_OP_GE 12
#define BINARY_OP_NE 15

OUT_COMPUTE_VEC4_T apply_binary_scalar_op(
    const int op_type,
    const COMPUTE_VEC4_T X,
    const COMPUTE_VEC4_T Y) {
  switch (op_type) {
#ifdef BINARY_SCALAR_IS_COMPARISON
    case BINARY_OP_NE:
      return OUT_COMPUTE_VEC4_T(notEqual(X, Y));
    case BINARY_OP_LT:
      return OUT_COMPUTE_VEC4_T(lessThan(X, Y));
    case BINARY_OP_LE:
      return OUT_COMPUTE_VEC4_T(lessThanEqual(X, Y));
    case BINARY_OP_GT:
      return OUT_COMPUTE_VEC4_T(greaterThan(X, Y));
    case BINARY_OP_GE:
      return OUT_COMPUTE_VEC4_T(greaterThanEqual(X, Y));
    default: // BINARY_OP_EQ
      return OUT_COMPUTE_VEC4_T(equal(X, Y));
#else
    case BINARY_OP_POW:
      return OUT_COMPUTE_VEC4_T(power_of(X, Y));
    default: // BINARY_OP_MUL
      return OUT_COMPUTE_VEC4_T(X * Y);
#endif
  }
}

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

  OUT_COMPUTE_VEC4_T out_texel = apply_binary_scalar_op(
      op_type, COMPUTE_VEC4_T(in_texel), COMPUTE_VEC4_T(scalar_value));

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
