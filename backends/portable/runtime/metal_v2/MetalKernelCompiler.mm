/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalStream.h"
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
  if (binaryArchive_) {
    [binaryArchive_ release];
  }
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Compiler_) {
      [mtl4Compiler_ release];
      mtl4Compiler_ = nil;
    }
  }
#endif
  [device_ release];
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
        binaryArchive_ = archive;
        [binaryArchive_ retain];
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
      [binaryArchive_ retain];
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

MetalKernel* MetalKernelCompiler::compile(
    const char* source,
    const char* functionName) {
  // Cache key includes a hash of the source so different sources reusing
  // the same function name (e.g., AOTI's "generated_kernel") don't collide.
  std::string key = std::to_string(std::hash<std::string_view>{}(
                        std::string_view(source))) +
      "/" + functionName;
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second.get();
  }

  @autoreleasepool {
    NSString* sourceStr = [NSString stringWithUTF8String:source];
    NSError* error = nil;

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

    // Metal 4: Use precise math mode
#if ET_METAL4_AVAILABLE
    if (@available(macOS 15.0, iOS 18.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    }
#endif

    id<MTLLibrary> library = [device_ newLibraryWithSource:sourceStr options:options error:&error];
    [options release];

    if (!library || error) {
      ET_LOG(Error, "MetalKernelCompiler: failed to compile shader: %s",
             error ? [[error localizedDescription] UTF8String] : "unknown");
      return nullptr;
    }

    NSString* funcName = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:funcName];

    if (!function) {
      ET_LOG(Error, "MetalKernelCompiler: function '%s' not found", functionName);
      [library release];
      return nullptr;
    }

#if ET_METAL4_ENABLE
    // ----- Metal 4 dispatch path -----
    // Use MTL4Compiler when available. The output is still id<MTLComputePipelineState>
    // which is the same protocol used by both legacy and MTL4 encoders.
    if (useMTL4()) {
      if (@available(macOS 26.0, iOS 26.0, *)) {
        if (!mtl4Compiler_) {
          MTL4CompilerDescriptor* compilerDesc = [[MTL4CompilerDescriptor alloc] init];
          NSError* compilerErr = nil;
          mtl4Compiler_ = [device_ newCompilerWithDescriptor:compilerDesc error:&compilerErr];
          [compilerDesc release];
          if (!mtl4Compiler_ || compilerErr) {
            ET_LOG(Error, "MetalKernelCompiler: MTL4Compiler creation failed: %s",
                   compilerErr ? [[compilerErr localizedDescription] UTF8String] : "unknown");
            // Fall through to legacy path
          }
        }

        if (mtl4Compiler_) {
          MTL4LibraryFunctionDescriptor* funcDesc = [[MTL4LibraryFunctionDescriptor alloc] init];
          funcDesc.name = funcName;
          funcDesc.library = library;

          MTL4ComputePipelineDescriptor* mtl4PipelineDesc = [[MTL4ComputePipelineDescriptor alloc] init];
          mtl4PipelineDesc.computeFunctionDescriptor = funcDesc;
          mtl4PipelineDesc.label = funcName;
          mtl4PipelineDesc.supportIndirectCommandBuffers = MTL4IndirectCommandBufferSupportStateEnabled;

          NSError* mtl4Err = nil;
          id<MTLComputePipelineState> mtl4Pipeline =
              [mtl4Compiler_ newComputePipelineStateWithDescriptor:mtl4PipelineDesc
                                              compilerTaskOptions:nil
                                                            error:&mtl4Err];
          [funcDesc release];
          [mtl4PipelineDesc release];

          if (mtl4Pipeline && !mtl4Err) {
            [function release];
            [library release];
            auto kernel = std::make_unique<MetalKernel>(mtl4Pipeline, functionName);
            [mtl4Pipeline release];
            MetalKernel* result = kernel.get();
            cache_[key] = std::move(kernel);
            ET_LOG(Info, "MetalKernelCompiler: compiled '%s' via MTL4Compiler", functionName);
            return result;
          }
          ET_LOG(Error, "MetalKernelCompiler: MTL4 pipeline creation failed for '%s': %s. Falling back to legacy.",
                 functionName,
                 mtl4Err ? [[mtl4Err localizedDescription] UTF8String] : "unknown");
          // Fall through to legacy path
        }
      }
    }
#endif

    // Create pipeline descriptor for binary archive support
    MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipelineDesc.computeFunction = function;
    pipelineDesc.label = funcName;
    pipelineDesc.supportIndirectCommandBuffers = YES;  // Enable ICB support

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

    // Compile using descriptor (required for ICB support)
    if (!pipeline) {
      pipeline = [device_ newComputePipelineStateWithDescriptor:pipelineDesc
                                                        options:MTLPipelineOptionNone
                                                     reflection:nil
                                                          error:&error];

      if (!pipeline || error) {
        ET_LOG(Error, "MetalKernelCompiler: failed to create pipeline: %s",
               error ? [[error localizedDescription] UTF8String] : "unknown");
        [function release];
        [library release];
        [pipelineDesc release];
        return nullptr;
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

    auto kernel = std::make_unique<MetalKernel>(pipeline, functionName);
    [pipeline release];

    MetalKernel* result = kernel.get();
    cache_[key] = std::move(kernel);

    ET_LOG(Info, "MetalKernelCompiler: compiled '%s'", functionName);
    return result;
  }
}

MetalKernel* MetalKernelCompiler::loadFromLibrary(
    const void* metallibData,
    size_t metallibSize,
    const char* functionName) {
  @autoreleasepool {
    NSError* error = nil;
    dispatch_data_t data = dispatch_data_create(
        metallibData, metallibSize, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    id<MTLLibrary> library = [device_ newLibraryWithData:data error:&error];
    dispatch_release(data);

    if (!library || error) {
      ET_LOG(Error, "MetalKernelCompiler: failed to load metallib: %s",
             error ? [[error localizedDescription] UTF8String] : "unknown");
      return nullptr;
    }

    NSString* funcName = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    [library release];

    if (!function) {
      ET_LOG(Error, "MetalKernelCompiler: function '%s' not found in metallib", functionName);
      return nullptr;
    }

    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (!pipeline || error) {
      return nullptr;
    }

    auto kernel = std::make_unique<MetalKernel>(pipeline, functionName);
    [pipeline release];

    std::string key = std::string("lib:") + functionName;
    MetalKernel* result = kernel.get();
    cache_[key] = std::move(kernel);

    return result;
  }
}


} // namespace metal_v2
} // namespace backends
} // namespace executorch

