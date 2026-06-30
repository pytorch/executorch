/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define BUFFER_VEC4_T ${texel_load_type(DTYPE, "buffer")}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
// freqs_cis is always bound as a buffer so the shader can flat-index it
// regardless of the caller's declared rank (2D [N, C] or 4D [1, N, C/2, 2]).
${layout_declare_tensor(B, "r", "t_freqs", DTYPE, "buffer", is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}

/*
 * Applies rotary positional embeddings to a tensor whose last dimension
 * contains pair-interleaved (real, imag) components. This matches EdgeTAM's
 * `apply_rotary_enc_without_complex` semantics, where the fused cos/sin
 * freqs tensor has a flattened [N, C] layout (cos, sin pairs interleaved).
 *
 * Inputs:
 *   t_in:    [B, N, C] (last dim packed as [r0, i0, r1, i1, ...])
 *   t_freqs: contiguous memory with N * C elements. May arrive at any rank
 *            (e.g. 2D [N, C] or 4D [1, N, C/2, 2]). Physically the values
 *            are [cos0, sin0, cos1, sin1, ...] along the final axis.
 *
 * Output:
 *   t_out: same shape as t_in
 *
 * Math per k in [0, C/2):
 *   out[2k]   = x[2k] * cos[k] - x[2k+1] * sin[k]
 *   out[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]
 *
 * Each thread processes one width-packed texel (4 elements = 2 (r, i) pairs).
 * All participating tensors are assumed to be width-packed with standard axis
 * maps.
 *
 * The freqs tensor is indexed using a flat (n_idx * C + c_offset) address to
 * remain correct regardless of input rank — the shape of t_freqs does not
 * need to match the logical [N, C] layout, only the underlying memory does.
 */
void main() {
  // Each thread computes one output texel of 4 elements along the last dim.
  TensorIndex4D out_tidx = zero_tensor4d_idx();
  out_tidx.data.x = int(gl_GlobalInvocationID.x) * 4;
  out_tidx.data.y = int(gl_GlobalInvocationID.y);
  out_tidx.data.z = int(gl_GlobalInvocationID.z);

  if (out_of_bounds(out_tidx, outp)) {
    return;
  }

  // Freqs tensor is always a contiguous buffer of N * C elements. Compute
  // a flat texel index directly from logical (n_idx, c_elem_idx / 4). The
  // logical width C comes from the output tensor metadata — both buffer
  // and texture metadata store this at index 0 (sizes[0][0] / sizes.x).
#ifdef USING_BUFFER
  const uint C_width = outp.sizes[0][0];
#else
  const uint C_width = uint(outp.sizes.x);
#endif
  const uint freqs_texel_bufi =
      uint(out_tidx.data.y) * div_4(C_width)
      + uint(gl_GlobalInvocationID.x);
  BUFFER_VEC4_T f_tex = t_freqs[freqs_texel_bufi];

#ifdef USING_BUFFER
  const uint out_texel_bufi =
      div_4(tensor4d_idx_to_linear_idx(outp, out_tidx));
  VEC4_T x_tex = t_in[out_texel_bufi];
#else // USING_TEXTURE
  const ivec3 out_pos =
      tensor4d_idx_to_texel_pos_simple(outp, out_tidx, outp_layout);
  VEC4_T x_tex = texelFetch(t_in, out_pos, 0);
#endif

  // x_tex = (r0, i0, r1, i1), f_tex = (c0, s0, c1, s1)
  VEC4_T out_tex;
  out_tex.x = x_tex.x * VEC4_T(f_tex).x - x_tex.y * VEC4_T(f_tex).y;
  out_tex.y = x_tex.x * VEC4_T(f_tex).y + x_tex.y * VEC4_T(f_tex).x;
  out_tex.z = x_tex.z * VEC4_T(f_tex).z - x_tex.w * VEC4_T(f_tex).w;
  out_tex.w = x_tex.z * VEC4_T(f_tex).w + x_tex.w * VEC4_T(f_tex).z;

#ifdef USING_BUFFER
  t_out[out_texel_bufi] = out_tex;
#else
  imageStore(t_out, out_pos, out_tex);
#endif
}
