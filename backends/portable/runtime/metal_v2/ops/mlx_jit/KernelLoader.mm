/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * KernelLoader.mm — implementation of the per-shape JIT kernel loaders.
 * Mirrors mlx/backend/metal/jit_kernels.cpp 1:1 for the four matmul
 * functions (dense fused / dense NAX / split-K / split-K NAX) plus the
 * shared split-K accum. Each function:
 *   1. assembles a small per-shape MSL source string by concatenating
 *      the relevant Snippets::* getters with one TemplateGen::makeInstantiation
 *      directive,
 *   2. hands the source + cache keys to MetalKernelCompiler::
 *      getOrCompilePsoFromSource (the vendor-neutral generic primitive).
 */

#include <executorch/backends/portable/runtime/metal_v2/ops/mlx_jit/KernelLoader.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/mlx_jit/Snippets.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/mlx_jit/TemplateGen.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit {

//===----------------------------------------------------------------------===//
// JitDtype helpers
//===----------------------------------------------------------------------===//

const char* typeToName(JitDtype dt) {
  switch (dt) {
    case JitDtype::Float32:  return "float32";
    case JitDtype::Float16:  return "float16";
    case JitDtype::BFloat16: return "bfloat16";
    case JitDtype::Bool:     return "bool_";  // matches MLX's bool_ name
  }
  return "unknown";
}

const char* typeToTemplateArg(JitDtype dt) {
  switch (dt) {
    case JitDtype::Float32:  return "float";
    case JitDtype::Float16:  return "float16_t";
    case JitDtype::BFloat16: return "bfloat16_t";
    case JitDtype::Bool:     return "bool";
  }
  return "float";
}

//===----------------------------------------------------------------------===//
// Singleton accessor
//===----------------------------------------------------------------------===//

namespace {
std::once_flag g_initOnce;
std::unique_ptr<KernelLoader> g_loader;
}

KernelLoader& shared(MetalKernelCompiler* compiler) {
  std::call_once(g_initOnce, [compiler]() {
    g_loader.reset(new KernelLoader(compiler));
  });
  return *g_loader;
}

void resetForTesting() {
  g_loader.reset();
  // Reset the once_flag by re-placement-constructing it.
  g_initOnce.~once_flag();
  new (&g_initOnce) std::once_flag();
}

//===----------------------------------------------------------------------===//
// getDenseGemmKernel — port of get_steel_gemm_fused_kernel.
// Source layout (jit_kernels.cpp:501-517):
//   metal::utils() << metal::gemm() << metal::steel_gemm_fused()
//   << get_template_definition(lib_name, "gemm", out_t, bm, bn, bk, wm, wn,
//                              transpose_a, transpose_b);
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getDenseGemmKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype outDtype,
    bool transpose_a, bool transpose_b,
    int bm, int bn, int bk, int wm, int wn) {
  // libraryCacheKey == kernelName so identical (tile, dtype, layout)
  // dispatches reuse the compiled MTLLibrary across PSOs (which differ
  // by FC values). PSO cache key includes hashName + fcs fingerprint
  // inside getOrCompilePsoFromSource.
  std::string psoKey = kernelName + "::" + hashName + fcs.fingerprint();
  // Source factory: only invoked on library cache miss.
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::utils() << Snippets::gemm() << Snippets::steel_gemm_fused()
        << TemplateGen::makeInstantiation(
               kernelName, "gemm",
               typeToTemplateArg(outDtype),
               bm, bn, bk, wm, wn,
               transpose_a ? "true" : "false",
               transpose_b ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

//===----------------------------------------------------------------------===//
// getDenseNaxKernel — port of get_steel_gemm_fused_nax_kernel.
// Source layout (jit_kernels.cpp:923-940):
//   metal::utils() << metal::gemm_nax() << metal::steel_gemm_fused_nax()
//   << get_template_definition(lib_name, "gemm", out_t, bm, bn, bk, wm, wn,
//                              transpose_a, transpose_b);
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getDenseNaxKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype outDtype,
    bool transpose_a, bool transpose_b,
    int bm, int bn, int bk, int wm, int wn) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::utils() << Snippets::gemm_nax()
        << Snippets::steel_gemm_fused_nax()
        << TemplateGen::makeInstantiation(
               kernelName, "gemm",
               typeToTemplateArg(outDtype),
               bm, bn, bk, wm, wn,
               transpose_a ? "true" : "false",
               transpose_b ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

//===----------------------------------------------------------------------===//
// getSplitKKernel — port of get_steel_gemm_splitk_kernel.
// Source layout (jit_kernels.cpp:536-554):
//   metal::utils() << metal::gemm() << metal::steel_gemm_splitk()
//   << get_template_definition(lib_name, "gemm_splitk", in_t, out_t,
//                              bm, bn, bk, wm, wn, transpose_a, transpose_b,
//                              mn_aligned, k_aligned);
// No FC tuple — alignment goes through template args.
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getSplitKKernel(
    const std::string& kernelName,
    JitDtype inDtype,
    JitDtype outDtype,
    bool transpose_a, bool transpose_b,
    int bm, int bn, int bk, int wm, int wn,
    bool mn_aligned, bool k_aligned) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::utils() << Snippets::gemm() << Snippets::steel_gemm_splitk()
        << TemplateGen::makeInstantiation(
               kernelName, "gemm_splitk",
               typeToTemplateArg(inDtype),
               typeToTemplateArg(outDtype),
               bm, bn, bk, wm, wn,
               transpose_a ? "true" : "false",
               transpose_b ? "true" : "false",
               mn_aligned ? "true" : "false",
               k_aligned ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

//===----------------------------------------------------------------------===//
// getSplitKNaxKernel — port of get_steel_gemm_splitk_nax_kernel.
// Source layout (jit_kernels.cpp:995-1010):
//   metal::utils() << metal::gemm_nax() << metal::steel_gemm_splitk_nax()
//   << get_template_definition(lib_name, "gemm_splitk_nax", out_t,
//                              bm, bn, bk, wm, wn, transpose_a, transpose_b);
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getSplitKNaxKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype outDtype,
    bool transpose_a, bool transpose_b,
    int bm, int bn, int bk, int wm, int wn) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::utils() << Snippets::gemm_nax()
        << Snippets::steel_gemm_splitk_nax()
        << TemplateGen::makeInstantiation(
               kernelName, "gemm_splitk_nax",
               typeToTemplateArg(outDtype),
               bm, bn, bk, wm, wn,
               transpose_a ? "true" : "false",
               transpose_b ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

//===----------------------------------------------------------------------===//
// getSplitKAccumKernel — port of get_steel_gemm_splitk_accum_kernel.
// Source layout (jit_kernels.cpp:566-577):
//   metal::utils() << metal::gemm() << metal::steel_gemm_splitk()
//   << get_template_definition(lib_name,
//                              axbpy ? "gemm_splitk_accum_axpby"
//                                    : "gemm_splitk_accum",
//                              in_t, out_t);
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getSplitKAccumKernel(
    const std::string& kernelName,
    JitDtype inDtype,
    JitDtype outDtype,
    bool axpby) {
  const char* func = axpby ? "gemm_splitk_accum_axpby" : "gemm_splitk_accum";
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::utils() << Snippets::gemm() << Snippets::steel_gemm_splitk()
        << TemplateGen::makeInstantiation(
               kernelName, func,
               typeToTemplateArg(inDtype),
               typeToTemplateArg(outDtype));
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

//===----------------------------------------------------------------------===//
// getGemvKernel / getGemvTKernel — JIT instantiation of MLX's gemv /
// gemv_t kernel templates from gemv.metal (vendored as Snippets::gemv()
// via the local-snippet wrapper that suppresses MLX's AOT
// instantiations).
// Template signature (gemv.metal:438 + 659):
//   <T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>
// where:
//   - T is the data type (float / float16_t / bfloat16_t)
//   - (BM, BN, SM, SN, TM, TN) are tile-shape ints
//   - kDoNCBatch / kDoAxpby are bool template args
// Source layout: just Snippets::gemv() — it already #includes both
// kernels/utils.h and kernels/steel/utils.h (transitively, via the
// thin local wrapper around gemv.metal). NO `gemm()` snippet
// concatenation needed (and including utils() too would cause
// redefinition errors since gemv() already inlines it).
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getGemvKernel(
    const std::string& kernelName,
    JitDtype dtype,
    int bm, int bn, int sm, int sn, int tm, int tn,
    bool nc_batch, bool axpby) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::gemv()
        << TemplateGen::makeInstantiation(
               kernelName, "gemv",
               typeToTemplateArg(dtype),
               bm, bn, sm, sn, tm, tn,
               nc_batch ? "true" : "false",
               axpby    ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getGemvTKernel(
    const std::string& kernelName,
    JitDtype dtype,
    int bm, int bn, int sm, int sn, int tm, int tn,
    bool nc_batch, bool axpby) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::gemv()
        << TemplateGen::makeInstantiation(
               kernelName, "gemv_t",
               typeToTemplateArg(dtype),
               bm, bn, sm, sn, tm, tn,
               nc_batch ? "true" : "false",
               axpby    ? "true" : "false");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

//===----------------------------------------------------------------------===//
// SDPA vector kernels (vendored from MLX 0.31.2's
// scaled_dot_product_attention.metal via the local sdpa_vector snippet).
// All three kernels share Snippets::sdpa_vector() — its body inlines
// kernels/utils.h transitively so we don't concatenate Snippets::utils()
// (which would cause redefinition errors).
//===----------------------------------------------------------------------===//

// getSdpaVectorKernel — single-pass `sdpa_vector<T, D, V>`.
// FCs (sdpa_vector.h:7-12, slots 20-25):
//   20 has_mask
//   21 query_transposed
//   22 do_causal
//   23 bool_mask
//   24 float_mask
//   25 has_sinks
// (slot 26 `blocks` is 2-pass-only)
id<MTLComputePipelineState> KernelLoader::getSdpaVectorKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype dtype, int D, int V) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::sdpa_vector()
        << TemplateGen::makeInstantiation(
               kernelName, "sdpa_vector",
               typeToTemplateArg(dtype), D, V);
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

// getSdpaVector2PassKernel1 — partial-accumulation pass.
// Same template family as sdpa_vector but with FC slot 26 (`blocks`) added.
id<MTLComputePipelineState> KernelLoader::getSdpaVector2PassKernel1(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype dtype, int D, int V) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::sdpa_vector()
        << TemplateGen::makeInstantiation(
               kernelName, "sdpa_vector_2pass_1",
               typeToTemplateArg(dtype), D, V);
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

// getSdpaVector2PassKernel2 — final aggregation. Template `<T, D>` only;
// no FCs (the 2-pass kernel reads `blocks` from buffer 4 here).
id<MTLComputePipelineState> KernelLoader::getSdpaVector2PassKernel2(
    const std::string& kernelName,
    JitDtype dtype, int D) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::sdpa_vector()
        << TemplateGen::makeInstantiation(
               kernelName, "sdpa_vector_2pass_2",
               typeToTemplateArg(dtype), D);
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Steel attention kernels (vendored from MLX 0.31.2's
// steel/attn/kernels/steel_attention{,_nax}.metal via local snippets).
// Template signature (steel_attention.h:60-69):
//   attention<T, BQ, BK, BD, WM, WN, MaskType=float, AccumType=float>
// Same shape for attention_nax (steel_attention_nax.h).
// FCs (slots 200/201/300/301/302):
//   200 align_Q   201 align_K
//   300 has_mask  301 do_causal  302 has_sinks
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getSteelAttentionKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype dtype, JitDtype maskDtype,
    int BQ, int BK, int BD, int WM, int WN) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::steel_attention()
        << TemplateGen::makeInstantiation(
               kernelName, "attention",
               typeToTemplateArg(dtype),
               BQ, BK, BD, WM, WN,
               typeToTemplateArg(maskDtype),
               "float");  // AccumType is always float per MLX
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

id<MTLComputePipelineState> KernelLoader::getSteelAttentionNaxKernel(
    const std::string& kernelName,
    const std::string& hashName,
    const MetalKernelCompiler::FunctionConstants& fcs,
    JitDtype dtype, JitDtype maskDtype,
    int BQ, int BK, int BD, int WM, int WN) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::steel_attention_nax()
        << TemplateGen::makeInstantiation(
               kernelName, "attention_nax",
               typeToTemplateArg(dtype),
               BQ, BK, BD, WM, WN,
               typeToTemplateArg(maskDtype),
               "float");
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      &fcs);
}

//===----------------------------------------------------------------------===//
// Affine quantized linear kernels (vendored from MLX 0.31.2's
// kernels/quantized.metal via the local quantized snippet).
// All four use Snippets::quantized() — its body inlines the necessary
// MLX headers (utils.h, steel/gemm/gemm.h, quantized_utils.h, quantized.h).
// No FCs in v0; all variants (alignment, sym/asym, batched) are baked
// into kernel name and template args.
// Kernel-name convention (matches the host_name we emit, not MLX's AOT
// naming):
//   affine_qmv_fast_<type>_gs<gs>_b<bits>_batch0
//   affine_qmv_<type>_gs<gs>_b<bits>_batch0
//   affine_qmv_quad_<type>_gs<gs>_b<bits>_d<D>_batch0
//   affine_qmm_t_<type>_gs<gs>_b<bits>_alN<0/1>_batch0
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> KernelLoader::getAffineQmvFastKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmv_fast",
               typeToTemplateArg(dtype),
               group_size, bits,
               "false");  // batched
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getAffineQmvKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmv",
               typeToTemplateArg(dtype),
               group_size, bits,
               "false");  // batched
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getAffineQmvQuadKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits, int D) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmv_quad",
               typeToTemplateArg(dtype),
               group_size, bits, D,
               "false");  // batched
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getAffineQmmTKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits, bool aligned_N) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmm_t",
               typeToTemplateArg(dtype),
               group_size, bits,
               aligned_N ? "true" : "false",  // aligned_N
               "false");                       // batched
               // BM=32, BK=32, BN=32 use template defaults
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getAffineQmmTSplitKKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits, bool aligned_N) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmm_t_splitk",
               typeToTemplateArg(dtype),
               group_size, bits,
               aligned_N ? "true" : "false");  // aligned_N
               // BM=32, BK=32, BN=32 use template defaults.
               // No `batched` template arg in qmm_t_splitk.
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

id<MTLComputePipelineState> KernelLoader::getAffineQmmTNaxKernel(
    const std::string& kernelName,
    JitDtype dtype, int group_size, int bits, bool aligned_N) {
  auto factory = [=]() {
    std::ostringstream src;
    src << Snippets::quantized_nax()
        << TemplateGen::makeInstantiation(
               kernelName, "affine_qmm_t_nax",
               typeToTemplateArg(dtype),
               group_size, bits,
               aligned_N ? "true" : "false",  // aligned_N
               "false",                        // batched
               64, 64, 64,                     // BM, BK, BN (override defaults)
               2, 2);                          // WM, WN
    return src.str();
  };
  return compiler_->getOrCompilePsoFromSource(
      /*libraryCacheKey=*/kernelName,
      factory,
      /*functionName=*/kernelName,
      /*constants=*/nullptr);
}

}  // namespace mlx_jit
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
