/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalStream.h"
#import "MetalMTL4Backend.h"  // for mtl4Backend_->getOrCreateMTL4Compiler()
#import "MetalKernelCache.h"
#include <executorch/runtime/platform/log.h>
#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalKernelCompiler
//===----------------------------------------------------------------------===//

MetalKernelCompiler::MetalKernelCompiler(id<MTLDevice> device) : device_(device), binaryArchive_(nil) {
  [device_ retain];
}

MetalKernelCompiler::~MetalKernelCompiler() {
  // mtl4Compiler_ moved to MetalMTL4Backend; backend owns
  // its lifetime. Only the binary archive needs cleanup here.
  if (binaryArchive_) {
    [binaryArchive_ release];
    binaryArchive_ = nil;
  }
  // No per-instance kernel cache to release — MetalKernelCache owns all
  // compiled kernels for the process lifetime.
}

bool MetalKernelCompiler::loadBinaryArchive(const char* path) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 11.0, iOS 14.0, *)) {
    @autoreleasepool {
      NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];

      MTLBinaryArchiveDescriptor* desc = [[MTLBinaryArchiveDescriptor alloc] init];
      desc.url = url;

      NSError* error = nil;
      id<MTLBinaryArchive> archive = [device_ newBinaryArchiveWithDescriptor:desc error:&error];
      [desc release];

      if (archive) {
        if (binaryArchive_) {
          [binaryArchive_ release];
        }
        // audit H7 fix: newBinaryArchiveWithDescriptor:error: already
        // returns +1 retained. The previous [binaryArchive_ retain] made
        // it +2; only one [release] later so the archive leaked.
        binaryArchive_ = archive;
        ET_LOG(Info, "MetalKernelCompiler: Loaded binary archive from %s", path);
        return true;
      } else {
        ET_LOG(Debug, "MetalKernelCompiler: No binary archive at %s: %s", path,
               error ? [[error localizedDescription] UTF8String] : "unknown");
      }
    }
  }
#endif
  return false;
}

bool MetalKernelCompiler::saveBinaryArchive(const char* path) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 11.0, iOS 14.0, *)) {
    if (!binaryArchive_) {
      // Create new archive if none exists
      MTLBinaryArchiveDescriptor* desc = [[MTLBinaryArchiveDescriptor alloc] init];
      NSError* error = nil;
      binaryArchive_ = [device_ newBinaryArchiveWithDescriptor:desc error:&error];
      [desc release];

      if (!binaryArchive_) {
        ET_LOG(Error, "MetalKernelCompiler: Failed to create binary archive");
        return false;
      }
      // audit H7 fix: newBinaryArchiveWithDescriptor: already returns +1.
      // No additional [retain] needed.
    }

    @autoreleasepool {
      NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
      NSError* error = nil;

      if ([binaryArchive_ serializeToURL:url error:&error]) {
        ET_LOG(Info, "MetalKernelCompiler: Saved binary archive to %s", path);
        return true;
      } else {
        ET_LOG(Error, "MetalKernelCompiler: Failed to save binary archive: %s",
               error ? [[error localizedDescription] UTF8String] : "unknown");
      }
    }
  }
#endif
  return false;
}

std::unique_ptr<MetalKernel> MetalKernelCompiler::compile(
    const char* source,
    const char* functionName,
    const FunctionConstants* constants) {
  // No caching here — caller (typically MetalKernelCache::findOrInsert)
  // is responsible for deduping. compile() is now a pure compile: source
  // → freshly built MetalKernel → unique_ptr returned. This makes it
  // safe to call concurrently from many threads racing the same key.

  @autoreleasepool {
    NSString* sourceStr = [NSString stringWithUTF8String:source];
    NSError* error = nil;

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

    // Metal 4: math mode. Match MLX's MSL build flags (-fno-fast-math but
    // NO Precise floating-point function override). Setting
    // mathFloatingPointFunctions=Precise forces the compiler to use the
    // strictest (slowest) implementations of transcendentals AND
    // restricts FMA fusion in matmul kernels — costing ~30% on Apple9+
    // NAX matmul vs MLX's default. We retain MathModeSafe (no-fast-math
    // for IEEE-correct numerics) but allow the compiler to pick its
    // standard (non-Precise) intrinsics.
#if ET_METAL4_AVAILABLE
    if (@available(macOS 15.0, iOS 18.0, *)) {
      options.mathMode = MTLMathModeSafe;
      // mathFloatingPointFunctions intentionally LEFT AT DEFAULT
      // (MTLMathFloatingPointFunctionsFast / unset) to match MLX's build.
    }
#endif

    id<MTLLibrary> library = [device_ newLibraryWithSource:sourceStr options:options error:&error];
    [options release];

    // audit H3 fix: previously `!library || error` rejected successful
    // compilations that emitted warnings (a non-nil NSError can accompany a
    // valid library object — Apple uses error to surface warnings as well
    // as failures). Only the null-result is a real failure; the error
    // object's content is logged for diagnostics either way.
    if (!library) {
      ET_LOG(Error, "MetalKernelCompiler: failed to compile shader: %s",
             error ? [[error localizedDescription] UTF8String] : "unknown");
      return nullptr;
    }
    if (error) {
      ET_LOG(Info, "MetalKernelCompiler: shader compiled with diagnostics: %s",
             [[error localizedDescription] UTF8String]);
      error = nil;  // reset for next API call
    }

    NSString* funcName = [NSString stringWithUTF8String:functionName];
    // Function constants: bind each (index, value) at PSO creation so the MSL
    // compiler can DCE branches that depend on them. The same kernel function
    // produces a different PSO per (source, name, constants) tuple — the
    // cache key above captures all three.
    // NOTE: fcv lifetime is intentionally hoisted outside the if-block so the
    // MTL4 path below can also pass it via funcDesc.constantValues. Without
    // this, MTL4 PSO creation fails for FC-aware kernels with the validation
    // error "function ... cannot be used to build a pipeline state. Use
    // newFunctionWithName:constantValues:..." (because the unspecialized
    // function reference is what gets used).
    id<MTLFunction> function = nil;
    MTLFunctionConstantValues* fcv = nil;
    if (constants && !constants->empty()) {
      fcv = [[MTLFunctionConstantValues alloc] init];
      for (auto& [idx, val] : constants->bools) {
        bool v = val;
        [fcv setConstantValue:&v type:MTLDataTypeBool atIndex:idx];
      }
      for (auto& [idx, val] : constants->ints) {
        int32_t v = val;
        [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:idx];
      }
      NSError* fcErr = nil;
      function = [library newFunctionWithName:funcName
                              constantValues:fcv
                                       error:&fcErr];
      if (!function) {
        ET_LOG(Error,
               "MetalKernelCompiler: failed to specialize '%s' with %zu "
               "bool + %zu int constants: %s",
               functionName, constants->bools.size(),
               constants->ints.size(),
               fcErr ? [[fcErr localizedDescription] UTF8String] : "unknown");
        [fcv release];
        [library release];
        return nullptr;
      }
    } else {
      function = [library newFunctionWithName:funcName];
    }

    if (!function) {
      ET_LOG(Error, "MetalKernelCompiler: function '%s' not found", functionName);
      [library release];
      return nullptr;
    }

#if ET_METAL4_ENABLE
    // ----- Metal 4 dispatch path -----
    // MTL4Compiler now lives on MetalMTL4Backend.
    // Get-or-create through the backend; falls back to legacy PSO
    // creation if the backend isn't wired or MTL4 isn't available.
    if (useMTL4() && mtl4Backend_) {
      if (@available(macOS 26.0, iOS 26.0, *)) {
        id<MTL4Compiler> mtl4Compiler = mtl4Backend_->getOrCreateMTL4Compiler();

        if (mtl4Compiler) {
          // MTL4 PSO creation path. NOTE: MTL4LibraryFunctionDescriptor does
          // not (currently) accept function-constant values directly. If the
          // caller supplied constants, the `function` above is already the
          // specialized MTLFunction (built via newFunctionWithName:constantValues:),
          // but MTL4LibraryFunctionDescriptor references functions by NAME
          // from the library — without the FC values, it'd resolve to the
          // unspecialized function and PSO creation would fail with
          // "function ... cannot be used to build a pipeline state".
          // Until MTL4 grows native FC support, FC-aware kernels skip the
          // MTL4 fast path and use the legacy PSO creation below (which uses
          // the already-specialized `function` directly). The legacy path
          // ICB support disabled — the ICB record/replay path was deleted in M6
          // (typed-setter migration). The legacy PSO creation below skips this
          // descriptor flag entirely, saving driver-side specialization cost.
          if (fcv) {
            ET_LOG(Info,
                   "MetalKernelCompiler: '%s' has function constants — "
                   "skipping MTL4 path (no FC support), using legacy.",
                   functionName);
            // Fall through to legacy path.
          } else {
          MTL4LibraryFunctionDescriptor* funcDesc = [[MTL4LibraryFunctionDescriptor alloc] init];
          funcDesc.name = funcName;
          funcDesc.library = library;

          MTL4ComputePipelineDescriptor* mtl4PipelineDesc = [[MTL4ComputePipelineDescriptor alloc] init];
          mtl4PipelineDesc.computeFunctionDescriptor = funcDesc;
          mtl4PipelineDesc.label = funcName;
          // M6/audit: supportIndirectCommandBuffers removed — ICB path is gone.

          NSError* mtl4Err = nil;
          id<MTLComputePipelineState> mtl4Pipeline =
              [mtl4Compiler newComputePipelineStateWithDescriptor:mtl4PipelineDesc
                                              compilerTaskOptions:nil
                                                            error:&mtl4Err];
          [funcDesc release];
          [mtl4PipelineDesc release];

          if (mtl4Pipeline && !mtl4Err) {
            [function release];
            [library release];
            if (fcv) [fcv release];
            auto kernel = std::make_unique<MetalKernel>(mtl4Pipeline, functionName);
            [mtl4Pipeline release];
            ET_LOG(Info, "MetalKernelCompiler: compiled '%s' via MTL4Compiler", functionName);
            return kernel;
          }
          ET_LOG(Error, "MetalKernelCompiler: MTL4 pipeline creation failed for '%s': %s. Falling back to legacy.",
                 functionName,
                 mtl4Err ? [[mtl4Err localizedDescription] UTF8String] : "unknown");
          // Fall through to legacy path
          }  // end !fcv branch
        }
      }
    }
#endif

    // Create pipeline descriptor for binary archive support.
    // M6/audit: supportIndirectCommandBuffers no longer set — the ICB
    // record/replay path was removed and PSOs no longer need to opt in to
    // ICB-compatible compilation (which carried a small per-PSO
    // specialization cost).
    MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipelineDesc.computeFunction = function;
    pipelineDesc.label = funcName;

    id<MTLComputePipelineState> pipeline = nil;

#if ET_METAL4_AVAILABLE
    // Try to load from binary archive first (fast path)
    if (@available(macOS 11.0, iOS 14.0, *)) {
      if (binaryArchive_) {
        // Try to get pre-compiled pipeline from archive
        MTLPipelineOption pipelineOptions = MTLPipelineOptionNone;
        pipeline = [device_ newComputePipelineStateWithDescriptor:pipelineDesc
                                                          options:pipelineOptions
                                                       reflection:nil
                                                            error:&error];

        if (pipeline) {
          ET_LOG(Debug, "MetalKernelCompiler: Loaded '%s' from binary archive", functionName);
        }
      }
    }
#endif

    // Compile using descriptor (lets us reuse binary-archive lookup above).
    if (!pipeline) {
      pipeline = [device_ newComputePipelineStateWithDescriptor:pipelineDesc
                                                        options:MTLPipelineOptionNone
                                                     reflection:nil
                                                          error:&error];

      if (!pipeline) {
        ET_LOG(Error, "MetalKernelCompiler: failed to create pipeline: %s",
               error ? [[error localizedDescription] UTF8String] : "unknown");
        [function release];
        [library release];
        [pipelineDesc release];
        return nullptr;
      }
      if (error) {
        ET_LOG(Info, "MetalKernelCompiler: pipeline created with diagnostics: %s",
               [[error localizedDescription] UTF8String]);
        error = nil;
      }

#if ET_METAL4_AVAILABLE
      // Add to binary archive for future use
      if (@available(macOS 11.0, iOS 14.0, *)) {
        if (binaryArchive_) {
          NSError* archiveError = nil;
          if ([binaryArchive_ addComputePipelineFunctionsWithDescriptor:pipelineDesc error:&archiveError]) {
            ET_LOG(Debug, "MetalKernelCompiler: Added '%s' to binary archive", functionName);
          }
        }
      }
#endif
    }

    [function release];
    [library release];
    [pipelineDesc release];
    if (fcv) [fcv release];

    auto kernel = std::make_unique<MetalKernel>(pipeline, functionName);
    [pipeline release];

    ET_LOG(Info, "MetalKernelCompiler: compiled '%s'", functionName);
    return kernel;
  }
}

//===----------------------------------------------------------------------===//
// Generic per-shape JIT primitive (vendor-neutral).
// Two-level cache:
//   1. libCache_  : libraryCacheKey -> MTLLibrary    (one MSL compile per
//                                                     unique source string)
//   2. psoCache_  : libraryCacheKey + "::" + functionName + fc.fingerprint
//                                                  -> MTLComputePipelineState
// The sourceFactory lambda is ONLY invoked on a libCache_ miss, so callers
// can build their per-shape source string lazily and not pay assembly cost
// when the library is already compiled.
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> MetalKernelCompiler::getOrCompilePsoFromSource(
    const std::string& libraryCacheKey,
    const std::function<std::string()>& sourceFactory,
    const std::string& functionName,
    const FunctionConstants* constants) {
  // ---- PSO cache check (fast path: hot dispatches hit here) ----
  // Routed through process-wide MetalKernelCache so PSOs compiled by ANY
  // thread are visible to all others. Single-thread single-process workloads
  // see no behavioral change; multi-thread workloads avoid duplicating
  // the warm-up cliff per worker.
  auto& shared = MetalKernelCache::shared();
  std::string psoKey = libraryCacheKey + "::" + functionName +
      (constants ? constants->fingerprint() : std::string{});
  if (auto cached = shared.findPso(psoKey); cached != nil) {
    return cached;
  }

  // ---- Library cache check / compile ----
  id<MTLLibrary> library = shared.findLibrary(libraryCacheKey);

  @autoreleasepool {
    NSError* error = nil;
    if (!library) {
      std::string source = sourceFactory();
      NSString* sourceStr = [NSString stringWithUTF8String:source.c_str()];

      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if ET_METAL4_AVAILABLE
      if (@available(macOS 15.0, iOS 18.0, *)) {
        // Same math-mode as the big-source compile() path: MathModeSafe
        // (no fast-math) but allow the compiler to pick its standard
        // (non-Precise) intrinsics. Required to match MLX's MSL build flags.
        options.mathMode = MTLMathModeSafe;
      }
#endif
      library = [device_ newLibraryWithSource:sourceStr options:options error:&error];
      [options release];

      if (!library) {
        ET_LOG(Error,
               "MetalKernelCompiler::getOrCompilePsoFromSource: failed to "
               "compile library '%s': %s",
               libraryCacheKey.c_str(),
               error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
      }
      if (error) {
        ET_LOG(Info,
               "MetalKernelCompiler::getOrCompilePsoFromSource: library '%s' "
               "compiled with diagnostics: %s",
               libraryCacheKey.c_str(),
               [[error localizedDescription] UTF8String]);
        error = nil;
      }
      // Hand the +1 retained library to the shared cache. If another thread
      // raced and inserted first, the cache releases ours and returns its
      // entry — we re-fetch to use the winning library.
      shared.insertLibrary(libraryCacheKey, library);
      library = shared.findLibrary(libraryCacheKey);
      ET_CHECK_MSG(library != nil,
                   "MetalKernelCache: insertLibrary lost library '%s'",
                   libraryCacheKey.c_str());
    }

    // ---- Function lookup (with optional FCs) ----
    NSString* funcName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = nil;
    MTLFunctionConstantValues* fcv = nil;
    if (constants && !constants->empty()) {
      fcv = [[MTLFunctionConstantValues alloc] init];
      for (auto& [idx, val] : constants->bools) {
        bool v = val;
        [fcv setConstantValue:&v type:MTLDataTypeBool atIndex:idx];
      }
      for (auto& [idx, val] : constants->ints) {
        int32_t v = val;
        [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:idx];
      }
      NSError* fcErr = nil;
      function = [library newFunctionWithName:funcName
                              constantValues:fcv
                                       error:&fcErr];
      [fcv release];
      if (!function) {
        ET_LOG(Error,
               "MetalKernelCompiler::getOrCompilePsoFromSource: failed to "
               "specialize '%s' in '%s' with %zu bool + %zu int constants: %s",
               functionName.c_str(), libraryCacheKey.c_str(),
               constants->bools.size(), constants->ints.size(),
               fcErr ? [[fcErr localizedDescription] UTF8String] : "unknown");
        return nil;
      }
    } else {
      function = [library newFunctionWithName:funcName];
      if (!function) {
        ET_LOG(Error,
               "MetalKernelCompiler::getOrCompilePsoFromSource: function '%s' "
               "not found in library '%s'",
               functionName.c_str(), libraryCacheKey.c_str());
        return nil;
      }
    }

    // ---- PSO creation ----
    NSError* psoErr = nil;
    id<MTLComputePipelineState> pso =
        [device_ newComputePipelineStateWithFunction:function error:&psoErr];
    [function release];

    if (!pso) {
      ET_LOG(Error,
             "MetalKernelCompiler::getOrCompilePsoFromSource: failed to create "
             "PSO for '%s' in '%s': %s",
             functionName.c_str(), libraryCacheKey.c_str(),
             psoErr ? [[psoErr localizedDescription] UTF8String] : "unknown");
      return nil;
    }

    // Hand the +1 retained PSO to the shared cache. As with library, if we
    // raced and lost, the cache consumes our +1 and we re-fetch the winner.
    shared.insertPso(psoKey, pso);
    pso = shared.findPso(psoKey);
    ET_CHECK_MSG(pso != nil,
                 "MetalKernelCache: insertPso lost PSO for '%s'",
                 psoKey.c_str());
    ET_LOG(Info,
           "MetalKernelCompiler::getOrCompilePsoFromSource: compiled '%s' in "
           "library '%s'",
           functionName.c_str(), libraryCacheKey.c_str());
    return pso;
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
