/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// mlx_jit::KernelLoader — port of MLX 0.31.2's per-shape JIT kernel loaders.
// One free-function singleton (mlx_jit::shared(compiler)) exposed to ops.
// The loader composes per-shape MSL source via Snippets + TemplateGen and
// hands the assembled string to MetalKernelCompiler::getOrCompilePsoFromSource
// (the vendor-neutral primitive).
// Mirrors:
//   mlx/backend/metal/jit_kernels.cpp:
//     get_steel_gemm_fused_kernel       → getDenseGemmKernel
//     get_steel_gemm_fused_nax_kernel   → getDenseNaxKernel
//     get_steel_gemm_splitk_kernel      → getSplitKKernel
//     get_steel_gemm_splitk_nax_kernel  → getSplitKNaxKernel
//     get_steel_gemm_splitk_accum_kernel → getSplitKAccumKernel
// The `kernelName` / `hashName` strings follow MLX's exact naming
// conventions (matmul.cpp:216-254) so the on-disk shader cache layout
// matches MLX's; this also means the [[host_name(...)]] in the JIT
// instantiation lines up with the lookup.
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

#import <Metal/Metal.h>

#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit {

// Element type names used in JIT kernel instantiation. Maps to MLX's
// get_type_string() output (kernel template arg) AND type_to_name() output
// (kernel name suffix).
// We support exactly the dtypes the MLX matmul path supports; callers
// (MatMulOp, AddMMOp) translate from executorch::aten::ScalarType.
// `Bool` is an SDPA-only mask dtype (steel attention has separate
// instantiations per (input_dtype, mask_dtype) pair where mask_dtype is
// either input_dtype or bool — see steel_attention.metal's
// instantiate_attn_mask_helper). Not used by matmul-family kernels.
enum class JitDtype {
  Float32,    // kernel arg "float",       name suffix "float32"
  Float16,    // kernel arg "float16_t",   name suffix "float16"
  BFloat16,   // kernel arg "bfloat16_t",  name suffix "bfloat16"
  Bool,       // kernel arg "bool",        name suffix "bool_" (mask only)
};

// MLX-style kernel-name suffix (e.g. "float32"). Matches type_to_name().
const char* typeToName(JitDtype dt);

// MLX-style kernel template arg (e.g. "float"). Matches get_type_string().
const char* typeToTemplateArg(JitDtype dt);

class KernelLoader {
 public:
  explicit KernelLoader(MetalKernelCompiler* compiler) : compiler_(compiler) {}

  // ---- Dense fused GEMM (SIMD-MMA, Apple7+) ----
  // Mirrors MLX's get_steel_gemm_fused_kernel + matmul.cpp's call site.
  // `kernelName` is the MLX library cache key + the [[host_name]] in the
  // single emitted template instantiation; e.g.
  //   "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"
  // `hashName` distinguishes PSO cache entries within the same library
  // when only function-constant values differ; e.g. the kernelName plus
  //   "_has_batch_n_use_out_source_n_..."
  // `fcs` is the FC tuple bound at PSO creation (slots 10/100/110/200/201/202).
  id<MTLComputePipelineState> getDenseGemmKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype outDtype,
      bool transpose_a, bool transpose_b,
      int bm, int bn, int bk, int wm, int wn);

  // ---- Dense fused GEMM (NAX, Apple9+) ----
  // Mirrors MLX's get_steel_gemm_fused_nax_kernel.
  id<MTLComputePipelineState> getDenseNaxKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype outDtype,
      bool transpose_a, bool transpose_b,
      int bm, int bn, int bk, int wm, int wn);

  // ---- Split-K GEMM partial (SIMD-MMA) ----
  // Mirrors MLX's get_steel_gemm_splitk_kernel. inDtype / outDtype usually
  // differ — partials accumulate into fp32 output regardless of in-dtype.
  id<MTLComputePipelineState> getSplitKKernel(
      const std::string& kernelName,
      JitDtype inDtype,
      JitDtype outDtype,
      bool transpose_a, bool transpose_b,
      int bm, int bn, int bk, int wm, int wn,
      bool mn_aligned, bool k_aligned);

  // ---- Split-K GEMM partial (NAX) ----
  // Mirrors MLX's get_steel_gemm_splitk_nax_kernel.
  id<MTLComputePipelineState> getSplitKNaxKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype outDtype,
      bool transpose_a, bool transpose_b,
      int bm, int bn, int bk, int wm, int wn);

  // ---- Split-K accum (shared by SIMD and NAX paths) ----
  // Mirrors MLX's get_steel_gemm_splitk_accum_kernel. Picks either
  // gemm_splitk_accum or gemm_splitk_accum_axpby based on `axpby`.
  id<MTLComputePipelineState> getSplitKAccumKernel(
      const std::string& kernelName,
      JitDtype inDtype,
      JitDtype outDtype,
      bool axpby);

  // ---- GEMV (mat[M,K] @ vec[K] = out[M]) ----
  // Mirrors MLX's gemv kernel template (gemv.metal:438-508). Vendored
  // via the gemv() snippet (which thinly wraps gemv.metal with the AOT
  // instantiation macros suppressed).
  // `kernelName` is the [[host_name]] for this shape; e.g.
  //   "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
  // matches MLX's instantiate_gemv_helper macro naming so JIT artifact
  // names line up with MLX's AOT names.
  // Template args: T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby.
  // No FCs — alignment / batch-stride layout are template args.
  id<MTLComputePipelineState> getGemvKernel(
      const std::string& kernelName,
      JitDtype dtype,
      int bm, int bn, int sm, int sn, int tm, int tn,
      bool nc_batch, bool axpby);

  // ---- GEMV transposed (mat[K,N] @ vec[K] = out[N]) ----
  // Mirrors MLX's gemv_t kernel template (gemv.metal:659-760). Same
  // ABI / template signature as gemv. Used by MatMulOp's M==1 decode
  // path with mat=B, in_vec=A_row, out_vec=C.
  id<MTLComputePipelineState> getGemvTKernel(
      const std::string& kernelName,
      JitDtype dtype,
      int bm, int bn, int sm, int sn, int tm, int tn,
      bool nc_batch, bool axpby);

  // ---- SDPA vector (single-pass) ----
  // Mirrors MLX's sdpa_vector kernel (sdpa_vector.h:15-177). Used for
  // decode (qL <= 8) when the kL/gqa heuristic doesn't trigger 2-pass.
  // Template signature: sdpa_vector<T, D, V>
  // FCs (slots 20-25): has_mask, query_transposed, do_causal,
  //                    bool_mask, float_mask, has_sinks
  // `kernelName` follows MLX's naming (scaled_dot_product_attention.cpp:
  // 343-348): "sdpa_vector_<get_type_string>_<D>_<V>"
  id<MTLComputePipelineState> getSdpaVectorKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype dtype, int D, int V);

  // ---- SDPA vector 2-pass: pass 1 (per-block partial) ----
  // Mirrors MLX's sdpa_vector_2pass_1 (sdpa_vector.h:179-318). Used for
  // long-kL decode where partial-then-aggregate beats single-pass.
  // Template signature: sdpa_vector_2pass_1<T, D, V>
  // FCs (slots 20-26): has_mask, query_transposed, do_causal,
  //                    bool_mask, float_mask, has_sinks, blocks
  // Note slot 26 (`blocks`) is an int FC unique to pass 1 — pass 2 reads
  // `blocks` from buffer 4 instead. We model it via a dedicated
  // `intConstants` extension to FunctionConstants — until that ships,
  // pass `blocks` here as a separate int and the loader bakes it into
  // the kernel name (so distinct `blocks` values yield distinct PSOs).
  // The MetalKernelCompiler currently supports only bool FCs, but a future
  // change will add int FC support. For now the impl uses bool FCs
  // 20-25 and bakes `blocks` into a hashName suffix (see impl).
  id<MTLComputePipelineState> getSdpaVector2PassKernel1(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype dtype, int D, int V);

  // ---- SDPA vector 2-pass: pass 2 (aggregation) ----
  // Mirrors MLX's sdpa_vector_2pass_2 (sdpa_vector.h:320-394). No FCs.
  // Template signature: sdpa_vector_2pass_2<T, D>  (D = value_dim only)
  // Kernel name format: "sdpa_vector_2pass_2_<get_type_string>_<V>"
  id<MTLComputePipelineState> getSdpaVector2PassKernel2(
      const std::string& kernelName,
      JitDtype dtype, int D);

  // ---- Steel attention (SIMD-MMA, Apple7-Apple8 + Apple9 fallback) ----
  // Mirrors MLX's get_steel_attention_kernel.
  // Template signature: attention<T, BQ, BK, BD, WM, WN, MaskType, float>
  //   T        — input dtype
  //   MaskType — same as T for float-mask; bool for bool-mask
  // FCs (slots 200/201/300/301/302):
  //   200 align_Q   201 align_K
  //   300 has_mask  301 do_causal  302 has_sinks
  // `kernelName` follows MLX naming (scaled_dot_product_attention.cpp:
  // 222-238): "steel_attention_<type_to_name(T)>_bq<BQ>_bk<BK>_bd<BD>"
  //           "_wm<WM>_wn<WN>_mask<type_to_name(MaskType)>"
  // `hashName` = kernelName + "_align_Q_<t/n>_align_K_<t/n>_has_mask_<t/n>"
  //              + "_do_causal_<t/n>_has_sinks_<t/n>"
  id<MTLComputePipelineState> getSteelAttentionKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype dtype, JitDtype maskDtype,
      int BQ, int BK, int BD, int WM, int WN);

  // ---- Steel attention NAX (Apple9+ cooperative-tensor matmul) ----
  // Mirrors MLX's get_steel_attention_nax_kernel.
  // Same template / FC layout as getSteelAttentionKernel above except the
  // template is `attention_nax<...>` (uses NAX MMA helpers from
  // steel/attn/nax.h).
  id<MTLComputePipelineState> getSteelAttentionNaxKernel(
      const std::string& kernelName,
      const std::string& hashName,
      const MetalKernelCompiler::FunctionConstants& fcs,
      JitDtype dtype, JitDtype maskDtype,
      int BQ, int BK, int BD, int WM, int WN);

  // ---- Affine quantized linear: qmv_fast (decode fast path) ----
  // Mirrors MLX's affine_qmv_fast kernel template (quantized.h:1495+).
  //   Template signature: affine_qmv_fast<T, group_size, bits, batched=false>
  //   No FCs in v0 (always non-batched; alignment baked into `_fast` selection).
  // Used when M=1 AND `K % (32 * group_size) == 0`.
  id<MTLComputePipelineState> getAffineQmvFastKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits);

  // ---- Affine quantized linear: qmv (decode generic) ----
  // Mirrors MLX's affine_qmv kernel template (quantized.h:1547+).
  //   Template signature: affine_qmv<T, group_size, bits, batched=false>
  // Used when M=1 AND the qmv_fast alignment doesn't hold.
  id<MTLComputePipelineState> getAffineQmvKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits);

  // ---- Affine quantized linear: qmv_quad (decode D-specialized) ----
  // Mirrors MLX's affine_qmv_quad kernel template (quantized.h:1443+).
  //   Template signature: affine_qmv_quad<T, group_size, bits, D, batched=false>
  // Used when M=1 AND bits==4 AND D ∈ {64, 128} (per MLX's instantiation table).
  id<MTLComputePipelineState> getAffineQmvQuadKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits, int D);

  // ---- Affine quantized linear: qmm_t (prefill, weight-transposed) ----
  // Mirrors MLX's affine_qmm_t kernel template (quantized.h:1086+).
  //   Template signature:
  //     affine_qmm_t<T, group_size, bits, aligned_N, batched=false,
  //                  BM=32, BK=32, BN=32>
  // Used when M > 1. `aligned_N = (N % BN == 0)` skips N-tail safe-loads.
  id<MTLComputePipelineState> getAffineQmmTKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits, bool aligned_N);

  // ---- Affine quantized linear: qmm_t_splitk (prefill, B==1 split-K) ----
  // Mirrors MLX's affine_qmm_t_splitk kernel template (quantized.h:1780+).
  //   Template signature:
  //     affine_qmm_t_splitk<T, group_size, bits, aligned_N,
  //                         BM=32, BK=32, BN=32>
  // Buffer ABI: w(0), scales(1), biases(2), x(3), y(4), K(5), N(6), M(7),
  //             k_partition_size(8), split_k_partition_stride(9).
  // Grid: (n_tiles, m_tiles, split_k). Block: (32, 2, 2).
  // Output is the [split_k, M, N] intermediate; sum along axis 0 via
  // getSplitKAccumKernel to produce final [M, N].
  id<MTLComputePipelineState> getAffineQmmTSplitKKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits, bool aligned_N);

  // ---- Affine quantized linear: qmm_t_nax (prefill, NAX cooperative-tensor) ----
  // Mirrors MLX's affine_qmm_t_nax kernel template (quantized_nax.h:1194+).
  //   Template signature:
  //     affine_qmm_t_nax<T, group_size, bits, aligned_N, batched=false,
  //                      BM=64, BK=64, BN=64, WM=2, WN=2>
  // Buffer ABI matches qmm_t (slots 0-7), with batched stride slots 8-15
  // template-DCE'd when batched=false.
  // Tile: BM=BN=64, BK=64 (overrides defaults 32). aligned_N = (N % 64 == 0).
  // Requires Apple9+ (MTLGPUFamilyApple9). K must be divisible by 64.
  id<MTLComputePipelineState> getAffineQmmTNaxKernel(
      const std::string& kernelName,
      JitDtype dtype, int group_size, int bits, bool aligned_N);

 private:
  MetalKernelCompiler* compiler_;  // not owned; must outlive this loader
};

// Process-wide singleton accessor. Lazy-init on first call: constructs a
// KernelLoader bound to the given compiler. Subsequent calls ignore the
// `compiler` arg and return the same instance — the compiler bound on
// the first call lives for the process lifetime alongside its singleton
// MetalStream.
// Tests can call resetForTesting() to wipe state so the next shared() call
// rebinds to a fresh compiler.
KernelLoader& shared(MetalKernelCompiler* compiler);

// Test-only: drop the singleton so the next shared() call rebinds.
void resetForTesting();

}  // namespace mlx_jit
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
