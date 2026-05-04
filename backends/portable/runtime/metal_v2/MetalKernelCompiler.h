/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalKernelCompiler — owns per-stream MSL → MTLLibrary → MTLComputePSO
// compilation. Two parallel APIs:
//   1. compile(source, funcName, fcs)        — big-source path. One MSL
//      string hosts many [[host_name(...)]] kernels; cache key includes
//      a hash of the source. Returns a MetalKernel wrapper. Local cache_
//      stores per-stream wrappers.
//   2. getOrCompilePsoFromSource(libKey, factory, funcName, fcs)
//                                            — per-shape JIT path. Library
//      and PSO live in the process-wide SharedPsoCache. Returns the PSO
//      directly (no MetalKernel wrapper). Used by mlx_jit/KernelLoader.
// MetalKernelCompiler also owns optional MTLBinaryArchive state and a
// borrowed MetalMTL4Backend* (so it can route MTL4 PSO creation through
// the backend on Apple silicon ≥ macOS 26).
// Extracted from MetalStream.h — definition unchanged. See SharedPsoCache.h
// for the process-wide PSO cache that getOrCompilePsoFromSource consults.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalKernel.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ET_METAL4_ENABLE — same compile-time gate used by MetalStream.h. We
// re-test it here because MetalKernelCompiler holds a conditionally-typed
// member (MetalMTL4Backend* mtl4Backend_).
#ifndef ET_METAL4_ENABLE
#define ET_METAL4_ENABLE 0
#endif

namespace executorch {
namespace backends {
namespace metal_v2 {

// Forward declaration so MetalKernelCompiler can hold a borrowed
// MetalMTL4Backend* without pulling in the full backend header.
class MetalMTL4Backend;

class MetalKernelCompiler {
public:
  explicit MetalKernelCompiler(id<MTLDevice> device);
  ~MetalKernelCompiler();

  // Optional MSL function-constant tuple passed at PSO creation time.
  // Each (index, value) is bound to the corresponding [[function_constant(N)]]
  // declaration in the shader source; the MSL compiler dead-code-eliminates
  // any branches that depend on these. Mirrors MLX's gemm pattern (e.g.
  // align_M / align_N / align_K / use_out_source / do_axpby in
  // mlx steel_gemm_fused.h:9-16). Different specializations of the same
  // (source, function name) live as separate cache entries — see fingerprint().
  // Bool FCs cover the matmul family. Int FCs were added for SDPA's
  // sdpa_vector_2pass_1 kernel (slot 26 = blocks; an int that the
  // 2-pass kernel reads in its inner loop). Same MTL binding mechanism
  // (newFunctionWithName:constantValues:) — just with MTLDataTypeInt at
  // the indicated slot.
  struct FunctionConstants {
    std::vector<std::pair<uint32_t, bool>> bools;
    std::vector<std::pair<uint32_t, int32_t>> ints;

    bool empty() const { return bools.empty() && ints.empty(); }

    // Short, deterministic suffix for the PSO cache key. Format:
    //   "@<idx><val>,..." for bools (val == '0'/'1')
    //   "#<idx>=<val>,..." for ints (val == decimal)
    // The two sections are kept separate so a 1-bit bool and a 1-int are
    // distinguishable in the cache key.
    std::string fingerprint() const {
      if (empty()) return std::string{};
      std::string s;
      if (!bools.empty()) {
        s += "@";
        for (auto& [idx, val] : bools) {
          s += std::to_string(idx);
          s += val ? '1' : '0';
          s += ',';
        }
      }
      if (!ints.empty()) {
        s += "#";
        for (auto& [idx, val] : ints) {
          s += std::to_string(idx);
          s += '=';
          s += std::to_string(val);
          s += ',';
        }
      }
      return s;
    }
  };

  // Returns an OWNED freshly-built MetalKernel. NO caching at this level —
  // the caller (typically via MetalKernelCache::findOrInsert) owns the
  // unique_ptr and decides whether to install it in the process-wide
  // cache. Returns nullptr on compile failure.
  std::unique_ptr<MetalKernel> compile(
      const char* source,
      const char* functionName,
      const FunctionConstants* constants = nullptr);

  //=== Generic per-shape JIT primitive (vendor-neutral) ===
  // Compile a small per-shape MSL source string into an MTLLibrary on cache
  // miss, then look up `functionName` and create a PSO. Both the library and
  // the PSO are cached (process-wide via SharedPsoCache); the source factory
  // is ONLY invoked on a library cache miss (lazy assembly).
  // Caller-chosen keys:
  //   * `libraryCacheKey` uniquely identifies the source string. Different
  //     callers must use different keys when their assembled sources differ;
  //     identical keys (e.g. same template tile params) reuse the same
  //     compiled MTLLibrary.
  //   * `functionName` is the Metal `[[host_name(...)]]` of the function
  //     to look up inside the compiled library.
  // FCs (optional): bound at function lookup time so the MSL compiler can
  // DCE FC-controlled branches at PSO creation. Different FC tuples produce
  // distinct PSOs (and distinct cache entries) for the same library +
  // function name pair.
  // No vendor-specific knowledge — used by ops/mlx_jit/ as the single
  // primitive for the per-shape JIT path.
  id<MTLComputePipelineState> getOrCompilePsoFromSource(
      const std::string& libraryCacheKey,
      const std::function<std::string()>& sourceFactory,
      const std::string& functionName,
      const FunctionConstants* constants = nullptr);

  //=== Binary Archive Support (Metal 4) ===

  /// Load binary archive from file (fast shader loading)
  bool loadBinaryArchive(const char* path);

  /// Save compiled shaders to binary archive
  bool saveBinaryArchive(const char* path);

  /// Check if binary archive is loaded
  bool hasBinaryArchive() const { return binaryArchive_ != nil; }

  // MTL4Compiler now lives on MetalMTL4Backend (the
  // architecturally-correct owner of MTL4-specific resources). MetalStream
  // calls this setter after constructing both objects so the kernel
  // compiler can route MTL4 PSO creation through the backend. Pre-R6.3
  // the field lived here; post-R6.3 we just hold a non-owning pointer.
  // Pointer is borrowed; the backend's lifetime exceeds the compiler's
  // (both owned by MetalStream; backend is destroyed AFTER compiler in
  // declaration order).
  // No-op when ET_METAL4_ENABLE=0 (the field doesn't exist; the MTL4
  // path in compile() is also dead-code-eliminated).
  void setMTL4Backend(MetalMTL4Backend* backend) {
#if ET_METAL4_ENABLE
    mtl4Backend_ = backend;
#else
    (void)backend;
#endif
  }

private:
  id<MTLDevice> device_;
  id<MTLBinaryArchive> binaryArchive_;
  // No per-instance kernel cache — MetalKernelCache (process-wide singleton)
  // is the canonical store. compile() returns owned kernels for the caller
  // to install via MetalKernelCache::findOrInsert (or to manage themselves
  // for one-shot uses like tests).

#if ET_METAL4_ENABLE
  // Borrowed pointer to the MTL4 backend (set via setMTL4Backend after
  // MetalStream finishes constructing both). nullptr when MTL4 isn't
  // available or hasn't been wired yet — compile() falls back to legacy
  // PSO creation in that case.
  MetalMTL4Backend* mtl4Backend_ = nullptr;
#endif
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
