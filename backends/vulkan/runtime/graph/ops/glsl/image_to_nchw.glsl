/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", BUF_DTYPE)}
${define_explicit_type_extensions(DTYPE)}

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

$if TO_STAGING:
  #define TO_STAGING
$if COALESCED_WRITES:
  #define COALESCED_WRITES

// The coalesced path writes contiguous NCHW staging offsets directly and does
// not honor buf_meta / buf_layout, so it is only valid for the staging
// destination. Fail the build if a coalesced non-staging (clone) variant is
// ever configured.
#if defined(COALESCED_WRITES) && !defined(TO_STAGING)
#error "COALESCED_WRITES requires TO_STAGING"
#endif

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_buffer(B, "w", "buf_out", BUF_DTYPE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "inp")}

$if not TO_STAGING:
  ${layout_declare_ubo(B, "BufferMetadata", "buf_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}

$if not TO_STAGING:
  ${layout_declare_spec_const(C, "int", "buf_layout", "CONTIG_LAYOUT_INT")}

void main() {
#ifdef COALESCED_WRITES
  // Output-centric dispatch: one thread per staging (NCHW) element. Consecutive
  // threads write consecutive staging offsets, so writes to the host-visible
  // staging buffer are fully coalesced. On a discrete GPU the staging buffer is
  // PCIe-backed, where coalescing the writes matters far more than coalescing
  // the (VRAM-cached) texture reads -- even though each texel is now fetched up
  // to 4 times, once per component. On unified-memory (mobile) GPUs the extra
  // fetches are a net loss, so this variant is gated to discrete GPUs.
  const int oi = int(gl_GlobalInvocationID.x);
  const int W = inp.sizes.x;
  const int H = inp.sizes.y;
  const int C = inp.sizes.z;
  const int N = inp.sizes.w;
  if (oi >= W * H * C * N) {
    return;
  }

  TensorIndex4D tidx;
  tidx.data.x = oi % W;
  tidx.data.y = (oi / W) % H;
  tidx.data.z = (oi / (W * H)) % C;
  tidx.data.w = oi / (W * H * C);

  const TextureElementIndex tex =
      tensor4d_idx_to_texture_element_idx_simple(inp, tidx, in_layout);
  #ifdef USING_TEXTURE2D
  const VEC4_T intex = texelFetch(t_in, tex.pos.xy, 0);
  #else
  const VEC4_T intex = texelFetch(t_in, tex.pos, 0);
  #endif
  buf_out[oi] = BUF_T(intex[tex.comp]);
#else
  // Texel-centric dispatch: one thread per texture texel, writing up to 4 packed
  // components. Reads are coalesced (one fetch per texel); writes are strided.
  // Preferred on unified-memory GPUs (mobile) where the staging buffer is not
  // PCIe-backed, so write coalescing buys nothing and the single fetch wins.
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(pos, inp)) {
    return;
  }

  TensorIndex4D tidx = texture_pos_to_tensor4d_idx_simple(inp, pos, in_layout);
  #ifdef USING_TEXTURE2D
  const VEC4_T intex = texelFetch(t_in, pos.xy, 0);
  #else
  const VEC4_T intex = texelFetch(t_in, pos, 0);
  #endif

  const int packed_dim = get_packed_dim(in_layout);
  int packed_dim_val;
  if (packed_dim == 0) {
    packed_dim_val = tidx.data.x;
  } else if (packed_dim == 1) {
    packed_dim_val = tidx.data.y;
  } else if (packed_dim == 2) {
    packed_dim_val = tidx.data.z;
  } else {
    packed_dim_val = tidx.data.w;
  }

  int packed_dim_size;
  if (packed_dim == 0) {
    packed_dim_size = inp.sizes.x;
  } else if (packed_dim == 1) {
    packed_dim_size = inp.sizes.y;
  } else if (packed_dim == 2) {
    packed_dim_size = inp.sizes.z;
  } else {
    packed_dim_size = inp.sizes.w;
  }

  int limit = min(4, packed_dim_size - packed_dim_val);

  for (int comp = 0; comp < limit; comp++) {
#ifdef TO_STAGING
    // Staging buffer is contiguous NCHW; compute the flat offset directly.
    int nchwi = tidx.data.x
              + tidx.data.y * inp.sizes.x
              + tidx.data.z * inp.sizes.x * inp.sizes.y
              + tidx.data.w * inp.sizes.x * inp.sizes.y * inp.sizes.z;
    buf_out[nchwi] = BUF_T(intex[comp]);
#else
    int bufi = tensor4d_idx_to_buf_idx(buf_meta, tidx, buf_layout);
    buf_out[bufi] = BUF_T(intex[comp]);
#endif

    if (packed_dim == 0) {
      tidx.data.x++;
    } else if (packed_dim == 1) {
      tidx.data.y++;
    } else if (packed_dim == 2) {
      tidx.data.z++;
    } else {
      tidx.data.w++;
    }
  }
#endif
}
