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
${layout_declare_tensor(B, "r", "t_freqs", DTYPE, "buffer", is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "freqs")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}

${buffer_scalar_type(DTYPE)} load_freq(const uint idx) {
  const TensorIndex tidx = contiguous_idx_to_tensor_idx(freqs, idx);
  const uint bufi = tensor_idx_to_linear_idx(freqs, tidx);
  return t_freqs[div_4(bufi)][bufi & 3u];
}

BUFFER_VEC4_T load_freqs(const uint idx) {
  const TensorIndex tidx = contiguous_idx_to_tensor_idx(freqs, idx);
  const uint bufi = tensor_idx_to_linear_idx(freqs, tidx);
  const BUFFER_VEC4_T first = t_freqs[div_4(bufi)];
  if (stride_at(freqs, 0) == 1u && (bufi & 3u) == 0u) {
    if (tidx.data[0].x + 3u < width(freqs)) {
      return first;
    }
    if (width(freqs) == 2u && tidx.data[0].x == 0u &&
        tidx.data[0].y + 1u < height(freqs) &&
        (stride_at(freqs, 1) & 3u) == 0u) {
      const BUFFER_VEC4_T second =
          t_freqs[div_4(bufi + stride_at(freqs, 1))];
      return BUFFER_VEC4_T(first.xy, second.xy);
    }
  }
  return BUFFER_VEC4_T(
      first[bufi & 3u],
      load_freq(idx + 1u),
      load_freq(idx + 2u),
      load_freq(idx + 3u));
}

/*
 * Applies rotary positional embeddings to a tensor whose last dimension
 * contains pair-interleaved (real, imag) components. This matches EdgeTAM's
 * `apply_rotary_enc_without_complex` semantics, where the fused cos/sin
 * freqs tensor has a flattened [N, C] layout (cos, sin pairs interleaved).
 *
 * Inputs:
 *   t_in:    [B, N, C] (last dim packed as [r0, i0, r1, i1, ...])
 *   t_freqs: N * C logical elements, with cos/sin pairs interleaved.
 *            Expected shapes: [N, C], [N, C/2, 2] or [1, N, C/2, 2].
 *
 * Output:
 *   t_out: same shape as t_in
 *
 * Math per k in [0, C/2):
 *   out[2k]   = x[2k] * cos[k] - x[2k+1] * sin[k]
 *   out[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]
 *
 * Each thread processes one width-packed texel (4 elements = 2 (r, i) pairs).
 * Input and output are width-packed with standard axis maps.
 *
 * Contiguous frequency quads use one vector load; two-wide frequency rows
 * use two vector loads and combine their xy lanes, excluding row padding.
 * Other layouts retain scalar addressing through buffer metadata.
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

#ifdef USING_BUFFER
  const uint C_width = outp.sizes[0][0];
#else
  const uint C_width = uint(outp.sizes.x);
#endif
  const uint freqs_idx =
      uint(out_tidx.data.y) * C_width + uint(out_tidx.data.x);
  const BUFFER_VEC4_T f_tex = load_freqs(freqs_idx);

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
