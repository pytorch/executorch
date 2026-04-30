/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// SdpaMlxJit — shared JIT-routing helpers for the SDPA family.
// Mirrors MatMulMlxJit.h's pattern. Each helper:
//   1. Picks tile sizes per MLX's heuristic (mlx/backend/metal/
//      scaled_dot_product_attention.cpp:166-326).
//   2. Builds the MLX-style kernel name + hash suffix.
//   3. Acquires a PSO via ops/mlx_jit/KernelLoader.
//   4. Binds the MLX-style buffer ABI and dispatches via MetalStream.
// Three dispatch helpers cover MLX's three SDPA paths:
//   - dispatchSdpaVectorViaMlxJit             (single-pass vector decode)
//   - dispatchSdpaVector2PassViaMlxJit        (2-pass vector for long kL)
//   - dispatchSteelAttentionViaMlxJit         (steel non-NAX or NAX prefill)
// Routing decisions (which helper to call) live in SDPAOp.mm.
// Host-side mirrors of MLX's AttnParams / AttnMaskParams structs are
// declared at the top with static_assert layout checks against MLX
// upstream sizes.
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/backends/portable/runtime/metal_v2/OpUtils.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/mlx_jit/KernelLoader.h>
#include <executorch/runtime/platform/log.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace sdpa_mlx_jit {

//===----------------------------------------------------------------------===//
// Host mirrors of MLX 0.31.2 structs (mlx/backend/metal/kernels/steel/attn/
// params.h).
//===----------------------------------------------------------------------===//

// Layout: 14 ints/floats (56B) followed by 4 × int64[3] arrays (96B) = 152B.
// All ints are 4B; alignment of int64[3] requires 8B alignment, satisfied by
// the 56B prefix being a multiple of 8.
struct AttnParamsHost {
  int B;            // batch size
  int H;            // number of query heads
  int D;            // head dim
  int qL;           // query sequence length
  int kL;           // key/value sequence length
  int gqa_factor;   // qH / kvH
  float scale;      // attention scale
  int NQ;           // number of query blocks (ceil(qL/BQ))
  int NK;           // number of K/V blocks
  int NQ_aligned;   // qL / BQ (truncated)
  int NK_aligned;   // kL / BK (truncated)
  int qL_rem;       // qL - NQ_aligned * BQ
  int kL_rem;       // kL - NK_aligned * BK
  int qL_off;       // kL - qL (causal sliding offset)
  int64_t Q_strides[3];  // Q strides over (B, H, L); D axis stride = 1
  int64_t K_strides[3];  // K strides over (B, H, L); D axis stride = 1
  int64_t V_strides[3];
  int64_t O_strides[3];
};
static_assert(sizeof(AttnParamsHost) == 152,
              "AttnParamsHost layout must match MLX upstream AttnParams "
              "(steel/attn/params.h)");

// Single int64[3] = 24 bytes.
struct AttnMaskParamsHost {
  int64_t M_strides[3];  // mask strides over (B, H, qL); kL axis stride = 1
};
static_assert(sizeof(AttnMaskParamsHost) == 24,
              "AttnMaskParamsHost layout must match MLX upstream "
              "AttnMaskParams");

//===----------------------------------------------------------------------===//
// Common utilities
//===----------------------------------------------------------------------===//

// Map ScalarType → JitDtype for SDPA inputs (fp32/fp16/bf16 only).
inline mlx_jit::JitDtype toJitDtype(executorch::aten::ScalarType dt) {
  switch (dt) {
    case executorch::aten::ScalarType::Float:    return mlx_jit::JitDtype::Float32;
    case executorch::aten::ScalarType::Half:     return mlx_jit::JitDtype::Float16;
    case executorch::aten::ScalarType::BFloat16: return mlx_jit::JitDtype::BFloat16;
    default:
      ET_CHECK_MSG(false, "sdpa_mlx_jit: unsupported dtype %d", int(dt));
      return mlx_jit::JitDtype::Float32;
  }
}

// MLX uses get_type_string for vector kernel names ("float", "float16_t",
// "bfloat16_t") — same as our JitDtype template arg name.
inline const char* vectorKernelTypeName(mlx_jit::JitDtype d) {
  return mlx_jit::typeToTemplateArg(d);  // "float" / "float16_t" / "bfloat16_t"
}

// Stride helpers for [B, H, L, D] tensors with D-axis stride == 1.
// MLX's sdpa_vector picks `head_stride = strides(0)` when shape(1) == 1 —
// covers MQA where K and V have a single shared head.
inline int64_t kvHeadStride(const executorch::aten::Tensor& t) {
  return (t.sizes()[1] == 1) ? t.strides()[0] : t.strides()[1];
}
inline int64_t seqStride(const executorch::aten::Tensor& t) {
  return t.strides()[2];
}

//===----------------------------------------------------------------------===//
// Kernel-name + hash builders (verbatim ports of MLX upstream).
//===----------------------------------------------------------------------===//

// MLX scaled_dot_product_attention.cpp:343-348 (single-pass) and 432-437
// (2-pass pass 1). Both share the same `sdpa_vector*_<type>_<D>_<V>` core.
inline std::string buildVectorKernelName(
    const char* kernel_kind,  // "sdpa_vector" or "sdpa_vector_2pass_1"
    mlx_jit::JitDtype dtype, int D, int V) {
  std::ostringstream s;
  s << kernel_kind << "_" << vectorKernelTypeName(dtype)
    << "_" << D << "_" << V;
  return s.str();
}

// MLX scaled_dot_product_attention.cpp:374-378 (single-pass).
inline std::string buildVectorHashSuffix(
    bool has_mask, bool bool_mask,
    bool query_transposed, bool do_causal, bool has_sinks) {
  std::ostringstream s;
  s << (has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask")
    << (query_transposed ? "_qt" : "_qnt")
    << (do_causal ? "_c" : "_nc")
    << (has_sinks ? "_sinks" : "_nosinks");
  return s.str();
}

// MLX scaled_dot_product_attention.cpp:518-522 (2-pass pass 1) — same
// suffix as single-pass but with a "_<blocks>" tail since `blocks` is an
// int FC and distinct values get distinct PSOs.
inline std::string buildVector2PassHashSuffix(
    bool has_mask, bool bool_mask,
    bool query_transposed, bool do_causal, bool has_sinks, int blocks) {
  std::ostringstream s;
  s << (has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask")
    << (query_transposed ? "_qt" : "_qnt")
    << (do_causal ? "_c" : "_nc")
    << (has_sinks ? "_sinks_" : "_nosinks_")
    << blocks;
  return s.str();
}

// MLX scaled_dot_product_attention.cpp:222-238.
inline std::string buildSteelKernelName(
    bool useNax,
    mlx_jit::JitDtype dtype, mlx_jit::JitDtype maskNameDtype,
    int BQ, int BK, int BD, int WM, int WN) {
  // Both NAX and non-NAX kernel names share the "steel_attention_" prefix;
  // routing happens inside the snippet (Snippets::steel_attention vs
  // steel_attention_nax) so only the LIBRARY differs, not the symbol.
  // MLX upstream uses the same prefix for both.
  (void)useNax;
  std::ostringstream s;
  s << "steel_attention_" << mlx_jit::typeToName(dtype)
    << "_bq" << BQ << "_bk" << BK << "_bd" << BD
    << "_wm" << WM << "_wn" << WN
    << "_mask" << mlx_jit::typeToName(maskNameDtype);
  return s.str();
}

// MLX scaled_dot_product_attention.cpp:240-253. align_Q / align_K vary by
// shape; has_mask / do_causal / has_sinks come from the call.
inline std::string buildSteelHashSuffix(
    bool align_Q, bool align_K,
    bool has_mask, bool do_causal, bool has_sinks) {
  std::ostringstream s;
  s << "_align_Q_" << (align_Q ? 't' : 'n')
    << "_align_K_" << (align_K ? 't' : 'n')
    << "_has_mask_" << (has_mask ? 't' : 'n')
    << "_do_causal_" << (do_causal ? 't' : 'n')
    << "_has_sinks_" << (has_sinks ? 't' : 'n');
  return s.str();
}

//===----------------------------------------------------------------------===//
// FC tuple builders.
//===----------------------------------------------------------------------===//

// Vector single-pass FC slots 20-25.
inline MetalKernelCompiler::FunctionConstants makeVectorFCs(
    bool has_mask, bool query_transposed, bool do_causal,
    bool bool_mask, bool float_mask, bool has_sinks) {
  return MetalKernelCompiler::FunctionConstants{
      /*bools=*/{
          {20, has_mask},
          {21, query_transposed},
          {22, do_causal},
          {23, bool_mask},
          {24, float_mask},
          {25, has_sinks},
      },
      /*ints=*/{},
  };
}

// Vector 2-pass-1 adds slot 26 (`blocks`, int).
inline MetalKernelCompiler::FunctionConstants makeVector2PassFCs(
    bool has_mask, bool query_transposed, bool do_causal,
    bool bool_mask, bool float_mask, bool has_sinks, int blocks) {
  return MetalKernelCompiler::FunctionConstants{
      /*bools=*/{
          {20, has_mask},
          {21, query_transposed},
          {22, do_causal},
          {23, bool_mask},
          {24, float_mask},
          {25, has_sinks},
      },
      /*ints=*/{
          {26, blocks},
      },
  };
}

// Steel attention FC slots 200/201/300/301/302.
inline MetalKernelCompiler::FunctionConstants makeSteelFCs(
    bool align_Q, bool align_K,
    bool has_mask, bool do_causal, bool has_sinks) {
  return MetalKernelCompiler::FunctionConstants{
      /*bools=*/{
          {200, align_Q},
          {201, align_K},
          {300, has_mask},
          {301, do_causal},
          {302, has_sinks},
      },
      /*ints=*/{},
  };
}

//===----------------------------------------------------------------------===//
// dispatchSdpaVectorViaMlxJit — single-pass vector kernel for short qL.
// Mirrors mlx/backend/metal/scaled_dot_product_attention.cpp::sdpa_vector
// (lines 329-416).
// Tensor shape contract (per MLX upstream):
//   Q: [B, Hq, qL, D]      with Q.strides[3] == 1
//   K: [B, Hkv, kL, D]     with K.strides[3] == 1
//   V: [B, Hkv, kL, V]     with V.strides[3] == 1
//   O: [B, Hq, qL, V]      contiguous output
//   mask (optional):       broadcast-compatible [..., qL, kL] (bool or T)
//   sinks (optional):      [Hq] per-head softmax bias (out of scope for v0)
// Buffer ABI (sdpa_vector.h:15-42):
//   0  Q                     11 bmask          (FC bool_mask)
//   1  K                     12 fmask          (FC float_mask)
//   2  V                     13 mask_kv_seq_stride int32 (FC has_mask)
//   3  O                     14 mask_q_seq_stride  int32 (FC has_mask)
//   4  gqa_factor int32      15 mask_head_stride   int32 (FC has_mask)
//   5  N int32               16 sinks            (FC has_sinks)
//   6  k_head_stride size_t  17 num_q_heads int32 (FC has_sinks)
//   7  k_seq_stride  size_t
//   8  v_head_stride size_t
//   9  v_seq_stride  size_t
//   10 scale float
// Grid: (B*Hq, qL, 1).  Block: (1024, 1, 1).
//===----------------------------------------------------------------------===//

inline void dispatchSdpaVectorViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& Q,
    const executorch::aten::Tensor& K,
    const executorch::aten::Tensor& V,
    const executorch::aten::Tensor* mask,  // nullable
    executorch::aten::Tensor& O,
    float scale,
    bool do_causal,
    executorch::aten::ScalarType dtype) {
  const auto jdt = toJitDtype(dtype);

  const int D = static_cast<int>(Q.sizes()[3]);
  const int Vdim = static_cast<int>(V.sizes()[3]);
  const int Hq = static_cast<int>(Q.sizes()[1]);
  const int Hkv = static_cast<int>(K.sizes()[1]);
  const int gqa_factor = Hq / Hkv;
  const int N = static_cast<int>(K.sizes()[2]);
  const int B = static_cast<int>(Q.sizes()[0]);
  const int qL = static_cast<int>(Q.sizes()[2]);

  const bool has_mask = (mask != nullptr);
  const bool bool_mask = has_mask &&
      mask->scalar_type() == executorch::aten::ScalarType::Bool;
  const bool float_mask = has_mask && !bool_mask;
  // Q layout: row_contiguous iff strides == default contiguous strides for
  // [B, Hq, qL, D]. Equivalent test: stride[2] == D AND stride[1] == qL*D
  // AND stride[0] == Hq*qL*D. We're conservative — assume not transposed
  // unless we can clearly detect it. MLX-side this comes from `flags()`.
  const int64_t expected_s2 = D;
  const int64_t expected_s1 = static_cast<int64_t>(qL) * D;
  const int64_t expected_s0 = static_cast<int64_t>(Hq) * qL * D;
  const bool query_row_contig =
      (Q.strides()[3] == 1) && (Q.strides()[2] == expected_s2) &&
      (Q.strides()[1] == expected_s1) && (Q.strides()[0] == expected_s0);
  const bool query_transposed = !query_row_contig;
  const bool has_sinks = false;  // sinks unsupported in v0

  const std::string kname =
      buildVectorKernelName("sdpa_vector", jdt, D, Vdim);
  const std::string hashName = kname + buildVectorHashSuffix(
      has_mask, bool_mask, query_transposed, do_causal, has_sinks);
  const auto fcs = makeVectorFCs(
      has_mask, query_transposed, do_causal, bool_mask, float_mask, has_sinks);

  ET_LOG(Debug,
         "dispatchSdpaVectorViaMlxJit: B=%d Hq=%d qL=%d D=%d V=%d kL=%d "
         "gqa=%d dtype=%s has_mask=%d (bool=%d) qt=%d causal=%d kname=%s",
         B, Hq, qL, D, Vdim, N, gqa_factor, dtypeSuffix(dtype),
         int(has_mask), int(bool_mask), int(query_transposed), int(do_causal),
         kname.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getSdpaVectorKernel(kname, hashName, fcs, jdt, D, Vdim);
  ET_CHECK_MSG(pso != nil,
               "dispatchSdpaVectorViaMlxJit: PSO acquisition failed for '%s'",
               kname.c_str());

  // Bind buffers.
  stream->setInput(0, Q.const_data_ptr(), Q.nbytes());
  stream->setInput(1, K.const_data_ptr(), K.nbytes());
  stream->setInput(2, V.const_data_ptr(), V.nbytes());
  stream->setOutput(3, O.mutable_data_ptr(), O.nbytes());
  stream->setBytes<int32_t>(4, gqa_factor);
  stream->setBytes<int32_t>(5, N);
  // Strides are size_t (8B) per kernel signature.
  stream->setBytes<int64_t>(6, kvHeadStride(K));
  stream->setBytes<int64_t>(7, seqStride(K));
  stream->setBytes<int64_t>(8, kvHeadStride(V));
  stream->setBytes<int64_t>(9, seqStride(V));
  stream->setBytes<float>(10, scale);
  if (has_mask) {
    // FC routes the kernel to read either bmask (slot 11) or fmask (slot 12).
    // Bind the mask buffer at the ACTIVE slot only (matches MLX which uses
    // `set_input_array(m, 11 + float_mask)`).
    const uint32_t mask_slot = bool_mask ? 11u : 12u;
    stream->setInput(mask_slot, mask->const_data_ptr(), mask->nbytes());
    // Mask strides: int32 per kernel signature.
    const int nd = mask->dim();
    int32_t kv_stride =
        (nd >= 1 && mask->sizes()[nd - 1] > 1) ? int32_t(mask->strides()[nd - 1]) : 0;
    int32_t q_stride =
        (nd >= 2 && mask->sizes()[nd - 2] > 1) ? int32_t(mask->strides()[nd - 2]) : 0;
    int32_t h_stride =
        (nd >= 3 && mask->sizes()[nd - 3] > 1) ? int32_t(mask->strides()[nd - 3]) : 0;
    stream->setBytes<int32_t>(13, kv_stride);
    stream->setBytes<int32_t>(14, q_stride);
    stream->setBytes<int32_t>(15, h_stride);
  }
  // sinks slots 16/17: FC-gated to has_sinks=false → unused; do not bind.

  uvec3 grid{uint32_t(B * Hq), uint32_t(qL), 1u};
  uvec3 block{1024u, 1u, 1u};
  stream->dispatch(pso, grid, block);
}

//===----------------------------------------------------------------------===//
// dispatchSdpaVector2PassViaMlxJit — 2-pass vector for long kL decode.
// Allocates 3 scratch buffers (intermediate / sums / maxs), runs pass 1 to
// fill them, then runs pass 2 to aggregate into `out`. Mirrors MLX upstream
// scaled_dot_product_attention.cpp::sdpa_vector_2pass (lines 418-584).
// Pass 1 buffer ABI (sdpa_vector.h:179-201):
//   0  Q          7  N int32                  15 mask_kv_seq_stride int32
//   1  K          8  k_head_stride size_t     16 mask_q_seq_stride  int32
//   2  V          9  k_seq_stride  size_t     17 mask_head_stride   int32
//   3  out=interm 10 v_head_stride size_t     18 sinks
//   4  sums       11 v_seq_stride  size_t
//   5  maxs       12 scale float
//                 13/14: bmask / fmask
// Pass 2 buffer ABI (sdpa_vector.h:320-330):
//   0 partials  1 sums  2 maxs  3 out  4 blocks(int32)
// Pass 1 grid:  (Hkv, B, blocks).  Block: (32, gqa, qL).
// Pass 2 grid:  (B*Hq, qL, 1).     Block: (1024, 1, 1).
// The `blocks` heuristic ports the device-class branching from MLX:
// scaled_dot_product_attention.cpp:443-476.
//===----------------------------------------------------------------------===//

// Compute MLX's `blocks` count for vector 2-pass. devc is the architecture
// suffix character: 's' → Apple7-8 (M-family), 'd' → Apple9 (M3+),
// otherwise → other (mostly Apple1-6 / iPad).
inline int chooseVector2PassBlocks(char devc, int N, int n_simds) {
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) blocks = 128;
      else if (N <= 32768) blocks = 256;
      else if (N <= 65536) blocks = 512;
      else blocks = 1024;
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) blocks = 512;
      else if (N >= 65536) blocks = 1024;
    }
  } else {
    blocks = (n_simds >= 4) ? 64 : 32;
  }
  return blocks;
}

inline void dispatchSdpaVector2PassViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& Q,
    const executorch::aten::Tensor& K,
    const executorch::aten::Tensor& V,
    const executorch::aten::Tensor* mask,  // nullable
    executorch::aten::Tensor& O,
    float scale,
    bool do_causal,
    executorch::aten::ScalarType dtype,
    char arch_suffix) {
  const auto jdt = toJitDtype(dtype);

  const int D = static_cast<int>(Q.sizes()[3]);
  const int Vdim = static_cast<int>(V.sizes()[3]);
  const int Hq = static_cast<int>(Q.sizes()[1]);
  const int Hkv = static_cast<int>(K.sizes()[1]);
  const int gqa_factor = Hq / Hkv;
  const int N = static_cast<int>(K.sizes()[2]);
  const int B = static_cast<int>(Q.sizes()[0]);
  const int qL = static_cast<int>(Q.sizes()[2]);

  const int n_simds = gqa_factor * qL;
  const int blocks = chooseVector2PassBlocks(arch_suffix, N, n_simds);

  const bool has_mask = (mask != nullptr);
  const bool bool_mask = has_mask &&
      mask->scalar_type() == executorch::aten::ScalarType::Bool;
  const bool float_mask = has_mask && !bool_mask;
  const int64_t expected_s2 = D;
  const int64_t expected_s1 = static_cast<int64_t>(qL) * D;
  const int64_t expected_s0 = static_cast<int64_t>(Hq) * qL * D;
  const bool query_row_contig =
      (Q.strides()[3] == 1) && (Q.strides()[2] == expected_s2) &&
      (Q.strides()[1] == expected_s1) && (Q.strides()[0] == expected_s0);
  const bool query_transposed = !query_row_contig;
  const bool has_sinks = false;

  // ---- Allocate scratch buffers ----
  // intermediate: [B, Hq, qL, blocks, V] in q.dtype()
  // sums / maxs : [B, Hq, qL, blocks]    in float32
  size_t element_size = 4;
  switch (dtype) {
    case executorch::aten::ScalarType::Half:
    case executorch::aten::ScalarType::BFloat16: element_size = 2; break;
    default: element_size = 4; break;
  }
  const size_t total_elems_per_thread = size_t(B) * Hq * qL * blocks;
  const size_t intermediate_bytes = total_elems_per_thread * Vdim * element_size;
  const size_t partials_bytes = total_elems_per_thread * sizeof(float);

  void* intermediate_ptr = stream->alloc(intermediate_bytes);
  void* sums_ptr = stream->alloc(partials_bytes);
  void* maxs_ptr = stream->alloc(partials_bytes);
  ET_CHECK_MSG(intermediate_ptr && sums_ptr && maxs_ptr,
               "dispatchSdpaVector2PassViaMlxJit: scratch allocation failed");

  // ---- Pass 1 ----
  const std::string p1_kname =
      buildVectorKernelName("sdpa_vector_2pass_1", jdt, D, Vdim);
  const std::string p1_hashName = p1_kname + buildVector2PassHashSuffix(
      has_mask, bool_mask, query_transposed, do_causal, has_sinks, blocks);
  const auto p1_fcs = makeVector2PassFCs(
      has_mask, query_transposed, do_causal, bool_mask, float_mask, has_sinks,
      blocks);

  ET_LOG(Debug,
         "dispatchSdpaVector2PassViaMlxJit: pass1 N=%d blocks=%d n_simds=%d "
         "kname=%s",
         N, blocks, n_simds, p1_kname.c_str());

  auto p1_pso = mlx_jit::shared(stream->compiler())
                    .getSdpaVector2PassKernel1(
                        p1_kname, p1_hashName, p1_fcs, jdt, D, Vdim);
  ET_CHECK_MSG(p1_pso != nil,
               "dispatchSdpaVector2PassViaMlxJit: pass1 PSO failed for '%s'",
               p1_kname.c_str());

  stream->setInput(0, Q.const_data_ptr(), Q.nbytes());
  stream->setInput(1, K.const_data_ptr(), K.nbytes());
  stream->setInput(2, V.const_data_ptr(), V.nbytes());
  stream->setOutput(3, intermediate_ptr, intermediate_bytes);
  stream->setOutput(4, sums_ptr, partials_bytes);
  stream->setOutput(5, maxs_ptr, partials_bytes);
  stream->setBytes<int32_t>(7, N);
  stream->setBytes<int64_t>(8, kvHeadStride(K));
  stream->setBytes<int64_t>(9, seqStride(K));
  stream->setBytes<int64_t>(10, kvHeadStride(V));
  stream->setBytes<int64_t>(11, seqStride(V));
  stream->setBytes<float>(12, scale);
  if (has_mask) {
    const uint32_t mask_slot = bool_mask ? 13u : 14u;
    stream->setInput(mask_slot, mask->const_data_ptr(), mask->nbytes());
    const int nd = mask->dim();
    int32_t kv_stride =
        (nd >= 1 && mask->sizes()[nd - 1] > 1) ? int32_t(mask->strides()[nd - 1]) : 0;
    int32_t q_stride =
        (nd >= 2 && mask->sizes()[nd - 2] > 1) ? int32_t(mask->strides()[nd - 2]) : 0;
    int32_t h_stride =
        (nd >= 3 && mask->sizes()[nd - 3] > 1) ? int32_t(mask->strides()[nd - 3]) : 0;
    stream->setBytes<int32_t>(15, kv_stride);
    stream->setBytes<int32_t>(16, q_stride);
    stream->setBytes<int32_t>(17, h_stride);
  }

  uvec3 p1_grid{uint32_t(Hkv), uint32_t(B), uint32_t(blocks)};
  uvec3 p1_block{32u, uint32_t(gqa_factor), uint32_t(qL)};
  stream->dispatch(p1_pso, p1_grid, p1_block);

  // ---- Pass 2 ----
  std::ostringstream p2_kn;
  p2_kn << "sdpa_vector_2pass_2_" << vectorKernelTypeName(jdt) << "_" << Vdim;
  const std::string p2_kname = p2_kn.str();

  ET_LOG(Debug,
         "dispatchSdpaVector2PassViaMlxJit: pass2 kname=%s blocks=%d",
         p2_kname.c_str(), blocks);

  auto p2_pso = mlx_jit::shared(stream->compiler())
                    .getSdpaVector2PassKernel2(p2_kname, jdt, Vdim);
  ET_CHECK_MSG(p2_pso != nil,
               "dispatchSdpaVector2PassViaMlxJit: pass2 PSO failed for '%s'",
               p2_kname.c_str());

  stream->setInput(0, intermediate_ptr, intermediate_bytes);
  stream->setInput(1, sums_ptr, partials_bytes);
  stream->setInput(2, maxs_ptr, partials_bytes);
  stream->setOutput(3, O.mutable_data_ptr(), O.nbytes());
  stream->setBytes<int32_t>(4, blocks);

  uvec3 p2_grid{uint32_t(B * Hq), uint32_t(qL), 1u};
  uvec3 p2_block{1024u, 1u, 1u};
  stream->dispatch(p2_pso, p2_grid, p2_block);

  // Scratch buffers must outlive the dispatch — schedule them for free
  // after the next sync. MetalStream::free() defers reclamation until the
  // hazard tracker confirms the GPU is done with the range.
  stream->free(intermediate_ptr);
  stream->free(sums_ptr);
  stream->free(maxs_ptr);
}

//===----------------------------------------------------------------------===//
// dispatchSteelAttentionViaMlxJit — steel (SIMD-MMA or NAX) prefill.
// Mirrors mlx/backend/metal/scaled_dot_product_attention.cpp::
//   sdpa_full_self_attention_metal (lines 166-327) for non-NAX
//   sdpa_full_self_attention_nax (lines 18-164) for NAX
// (the two are nearly identical; only the kernel + tile sizes differ).
// Tile selection:
//   non-NAX: WM=4, WN=1; BQ=32, BK=32 if D<128 else 16
//   NAX:     WM=4, WN=1; BQ=64, BK=32 if D<128 else 64
// Buffer ABI (steel_attention.h:71-78):
//   0  Q                  4  AttnParams      (constant)
//   1  K                  5  AttnMaskParams  (constant; FC has_mask)
//   2  V                  6  mask            (FC has_mask)
//   3  O                  7  sinks           (FC has_sinks)
// Grid: (NQ, H, B).  Block: (32, WM, WN).
//===----------------------------------------------------------------------===//

inline void dispatchSteelAttentionViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& Q,
    const executorch::aten::Tensor& K,
    const executorch::aten::Tensor& V,
    const executorch::aten::Tensor* mask,  // nullable
    executorch::aten::Tensor& O,
    float scale,
    bool do_causal,
    executorch::aten::ScalarType dtype,
    bool useNax) {
  const auto jdt = toJitDtype(dtype);

  const int B = static_cast<int>(Q.sizes()[0]);
  const int H = static_cast<int>(Q.sizes()[1]);
  const int D = static_cast<int>(Q.sizes()[3]);
  const int Hkv = static_cast<int>(K.sizes()[1]);
  const int gqa_factor = H / Hkv;
  const int qL = static_cast<int>(Q.sizes()[2]);
  const int kL = static_cast<int>(K.sizes()[2]);

  const int WM = 4, WN = 1;
  const int BD = D;
  const int BQ = useNax ? 64 : 32;
  // NAX: always BK=32 (per MLX upstream scaled_dot_product_attention.cpp::
  // sdpa_full_self_attention_nax — bk hardcoded to 32 regardless of D).
  // Non-NAX: BK=32 for small D, BK=16 for D>=128.
  const int BK = useNax ? 32 : (D < 128 ? 32 : 16);

  const bool align_Q = (qL % BQ) == 0;
  const bool align_K = (kL % BK) == 0;
  const bool has_mask = (mask != nullptr);
  const bool has_sinks = false;

  // Mask name dtype: per MLX, type_to_name(has_mask ? *mask : q). For
  // bool mask MLX uses "bool_". For float mask, same as input dtype.
  mlx_jit::JitDtype maskNameDtype = jdt;
  mlx_jit::JitDtype maskTypeArg = jdt;
  if (has_mask) {
    if (mask->scalar_type() == executorch::aten::ScalarType::Bool) {
      maskNameDtype = mlx_jit::JitDtype::Bool;
      maskTypeArg = mlx_jit::JitDtype::Bool;
    } else {
      maskNameDtype = jdt;
      maskTypeArg = jdt;
    }
  }

  const std::string kname = buildSteelKernelName(
      useNax, jdt, maskNameDtype, BQ, BK, BD, WM, WN);
  const std::string hashName = kname + buildSteelHashSuffix(
      align_Q, align_K, has_mask, do_causal, has_sinks);
  const auto fcs = makeSteelFCs(align_Q, align_K, has_mask, do_causal, has_sinks);

  ET_LOG(Debug,
         "dispatchSteelAttentionViaMlxJit: B=%d H=%d qL=%d kL=%d D=%d "
         "tile=(BQ=%d,BK=%d,BD=%d) useNax=%d dtype=%s causal=%d kname=%s",
         B, H, qL, kL, D, BQ, BK, BD, int(useNax),
         dtypeSuffix(dtype), int(do_causal), kname.c_str());

  id<MTLComputePipelineState> pso = nil;
  if (useNax) {
    pso = mlx_jit::shared(stream->compiler())
              .getSteelAttentionNaxKernel(
                  kname, hashName, fcs, jdt, maskTypeArg,
                  BQ, BK, BD, WM, WN);
  } else {
    pso = mlx_jit::shared(stream->compiler())
              .getSteelAttentionKernel(
                  kname, hashName, fcs, jdt, maskTypeArg,
                  BQ, BK, BD, WM, WN);
  }
  ET_CHECK_MSG(pso != nil,
               "dispatchSteelAttentionViaMlxJit: PSO failed for '%s'",
               kname.c_str());

  // Compose AttnParams.
  const int NQ = (qL + BQ - 1) / BQ;
  const int NK = (kL + BK - 1) / BK;
  const int NQ_aligned = qL / BQ;
  const int NK_aligned = kL / BK;

  AttnParamsHost params{
      /*B=*/B, /*H=*/H, /*D=*/D,
      /*qL=*/qL, /*kL=*/kL,
      /*gqa_factor=*/gqa_factor, /*scale=*/scale,
      /*NQ=*/NQ, /*NK=*/NK,
      /*NQ_aligned=*/NQ_aligned, /*NK_aligned=*/NK_aligned,
      /*qL_rem=*/(qL - NQ_aligned * BQ),
      /*kL_rem=*/(kL - NK_aligned * BK),
      /*qL_off=*/(kL - qL),
      /*Q_strides=*/{Q.strides()[0], Q.strides()[1], Q.strides()[2]},
      /*K_strides=*/{K.strides()[0], K.strides()[1], K.strides()[2]},
      /*V_strides=*/{V.strides()[0], V.strides()[1], V.strides()[2]},
      /*O_strides=*/{O.strides()[0], O.strides()[1], O.strides()[2]},
  };

  // Bind buffers.
  stream->setInput(0, Q.const_data_ptr(), Q.nbytes());
  stream->setInput(1, K.const_data_ptr(), K.nbytes());
  stream->setInput(2, V.const_data_ptr(), V.nbytes());
  stream->setOutput(3, O.mutable_data_ptr(), O.nbytes());
  stream->setBytes(4, &params, sizeof(params));
  if (has_mask) {
    AttnMaskParamsHost mp{
        /*M_strides=*/{
            mask->strides()[0], mask->strides()[1], mask->strides()[2]},
    };
    stream->setBytes(5, &mp, sizeof(mp));
    stream->setInput(6, mask->const_data_ptr(), mask->nbytes());
  }
  // sinks slot 7: FC-gated to has_sinks=false → unused; do not bind.

  uvec3 grid{uint32_t(NQ), uint32_t(H), uint32_t(B)};
  uvec3 block{32u, uint32_t(WM), uint32_t(WN)};
  stream->dispatch(pso, grid, block);
}

}  // namespace sdpa_mlx_jit
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
