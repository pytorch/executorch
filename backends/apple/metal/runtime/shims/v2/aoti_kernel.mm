/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI kernel-dispatch impl built directly on MetalStream.
//
// No ETMetalShaderLibrary / ETMetalKernelFunction / ETMetalStream classes.
// Opaque AOTI handles point to the two lightweight structs below.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_kernel.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

using metal_v2::Arg;
using metal_v2::MetalKernel;
using metal_v2::MetalStream;
using metal_v2::uvec3;

// =========================================================================
// Internal handle structs — not visible outside this file
// =========================================================================

// Holds shader source for deferred compilation. MetalKernelCompiler caches
// compiled kernels internally, so we just need the source string.
struct ShaderLibrary {
  std::string source;
};

// Holds a compiled kernel + accumulated args between start_encoding and
// dispatch.
struct KernelDispatcher {
  MetalKernel* kernel; // owned by MetalKernelCompiler cache
  ShaderLibrary* parent; // kept alive by library_storage

  struct DeferredArg {
    enum Kind { BUFFER, SCALAR } kind;
    void* ptr = nullptr;
    size_t size = 0;
    uint8_t scalar[8] = {};
  };
  std::vector<DeferredArg> args;

  void clear() { args.clear(); }

  void setBuffer(unsigned idx, void* ptr, size_t size) {
    if (idx >= args.size()) args.resize(idx + 1);
    args[idx] = {DeferredArg::BUFFER, ptr, size, {}};
  }

  void setScalarInt(unsigned idx, int64_t val) {
    if (idx >= args.size()) args.resize(idx + 1);
    auto& a = args[idx];
    a.kind = DeferredArg::SCALAR;
    a.size = sizeof(int64_t);
    std::memcpy(a.scalar, &val, sizeof(val));
  }

  // Flush accumulated args through MetalStream::dispatch().
  void dispatch(uvec3 grid, uvec3 block) {
    auto* stream = getMetalStream();

    // Register buffer args so MetalStream can map them to MTLBuffers.
    for (auto& a : args) {
      if (a.kind == DeferredArg::BUFFER && a.ptr) {
        stream->registerExternalBuffer(a.ptr, a.size);
      }
    }

    // Build Arg vector. MetalStream::dispatch takes initializer_list (Arg
    // has a private default ctor, so we can't default-construct an array
    // of them); build a std::vector via emplace_back, then dispatch via a
    // switch on the count.
    // TODO: add dispatch(kernel, span<Arg>, grid, block) to MetalStream.
    std::vector<Arg> gpuArgs;
    size_t n = std::min(args.size(), (size_t)8);
    gpuArgs.reserve(n);
    for (size_t i = 0; i < n; i++) {
      auto& a = args[i];
      if (a.kind == DeferredArg::BUFFER) {
        gpuArgs.emplace_back(a.ptr, a.size);
      } else {
        int64_t v;
        std::memcpy(&v, a.scalar, sizeof(v));
        gpuArgs.emplace_back(v);
      }
    }

    switch (n) {
      case 0: stream->dispatch(kernel, {}, grid, block); break;
      case 1: stream->dispatch(kernel, {gpuArgs[0]}, grid, block); break;
      case 2: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1]}, grid, block); break;
      case 3: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2]}, grid, block); break;
      case 4: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2], gpuArgs[3]}, grid, block); break;
      case 5: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2], gpuArgs[3], gpuArgs[4]}, grid, block); break;
      case 6: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2], gpuArgs[3], gpuArgs[4], gpuArgs[5]}, grid, block); break;
      case 7: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2], gpuArgs[3], gpuArgs[4], gpuArgs[5], gpuArgs[6]}, grid, block); break;
      case 8: stream->dispatch(kernel, {gpuArgs[0], gpuArgs[1], gpuArgs[2], gpuArgs[3], gpuArgs[4], gpuArgs[5], gpuArgs[6], gpuArgs[7]}, grid, block); break;
    }

    clear();
  }
};

// Lifetime management — owns the heap objects behind opaque handles.
static std::unordered_map<ShaderLibrary*, std::unique_ptr<ShaderLibrary>> library_storage;
static std::unordered_map<KernelDispatcher*, std::unique_ptr<KernelDispatcher>> function_storage;

// =========================================================================
// Grid/block helpers
// =========================================================================

static uvec3 computeBlock1D(MetalKernel* k, uint64_t length) {
  uint32_t max = k->maxThreadsPerThreadgroup().x;
  return uvec3(std::min(max, (uint32_t)length), 1, 1);
}

static uvec3 computeGrid1D(uint64_t length, uint32_t blockX) {
  return uvec3(((uint32_t)length + blockX - 1) / blockX, 1, 1);
}

// =========================================================================
// C-ABI shim implementations
// =========================================================================

extern "C" {

// --- Shader library ---

AOTITorchError aoti_torch_mps_create_shader_library(
    const char* source,
    AOTIMetalShaderLibraryHandle* out) {
  if (!source || !out) return Error::InvalidArgument;

  // DIAGNOSTIC: dump the shader source so we can see exactly what AOTI
  // generated for each kernel. Truncate to 1500 chars to keep logs sane.
  size_t src_len = std::strlen(source);
  ET_LOG(Info,
      "[shader-src] create_shader_library len=%zu source:\n----\n%.1500s%s\n----",
      src_len, source, src_len > 1500 ? "\n... [truncated]" : "");

  auto lib = std::make_unique<ShaderLibrary>();
  lib->source = source;
  auto* raw = lib.get();
  library_storage[raw] = std::move(lib);
  *out = reinterpret_cast<AOTIMetalShaderLibraryHandle>(raw);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle handle) {
  if (!handle) return Error::InvalidArgument;
  auto* lib = reinterpret_cast<ShaderLibrary*>(handle);
  library_storage.erase(lib);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle lib_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* out) {
  if (!lib_handle || !kernel_name || !out) return Error::InvalidArgument;

  auto* lib = reinterpret_cast<ShaderLibrary*>(lib_handle);
  MetalKernel* kernel = getMetalStream()->compiler()->compile(
      lib->source.c_str(), kernel_name);
  if (!kernel) {
    ET_LOG(Error, "Failed to compile kernel '%s'", kernel_name);
    return Error::Internal;
  }

  auto disp = std::make_unique<KernelDispatcher>();
  disp->kernel = kernel;
  disp->parent = lib;
  auto* raw = disp.get();
  function_storage[raw] = std::move(disp);
  *out = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw);
  return Error::Ok;
}

// --- Encoding / args / dispatch ---

AOTITorchError aoti_torch_mps_start_encoding(AOTIMetalKernelFunctionHandle func) {
  if (!func) return Error::InvalidArgument;
  reinterpret_cast<KernelDispatcher*>(func)->clear();
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func, unsigned idx, AOTITensorHandle tensor) {
  if (!func || !tensor) return Error::InvalidArgument;
  auto* t = reinterpret_cast<Tensor*>(tensor);
  reinterpret_cast<KernelDispatcher*>(func)->setBuffer(
      idx, t->data_ptr(), t->numel() * t->itemsize());
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func, unsigned idx, int64_t val) {
  if (!func) return Error::InvalidArgument;
  reinterpret_cast<KernelDispatcher*>(func)->setScalarInt(idx, val);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func, uint64_t length) {
  if (!func) return Error::InvalidArgument;
  auto* d = reinterpret_cast<KernelDispatcher*>(func);
  uvec3 block = computeBlock1D(d->kernel, length);
  d->dispatch(computeGrid1D(length, block.x), block);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func, uint64_t length, uint64_t group_size) {
  if (!func) return Error::InvalidArgument;
  auto* d = reinterpret_cast<KernelDispatcher*>(func);
  uint32_t max = d->kernel->maxThreadsPerThreadgroup().x;
  uint32_t bx = group_size > 0 ? std::min((uint32_t)group_size, max)
                                : std::min(max, (uint32_t)length);
  d->dispatch(computeGrid1D(length, bx), uvec3(bx, 1, 1));
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length, size_t length_size) {
  if (!func || !length || length_size == 0) return Error::InvalidArgument;
  auto* d = reinterpret_cast<KernelDispatcher*>(func);
  uint32_t max = d->kernel->maxThreadsPerThreadgroup().x;

  uvec3 grid, block;
  if (length_size == 1) {
    uint32_t bx = std::min(max, (uint32_t)length[0]);
    block = uvec3(bx, 1, 1);
    grid = uvec3(((uint32_t)length[0] + bx - 1) / bx, 1, 1);
  } else if (length_size == 2) {
    uint32_t bx = std::min(32u, (uint32_t)length[0]);
    uint32_t by = max / bx;
    block = uvec3(bx, by, 1);
    grid = uvec3(((uint32_t)length[0] + bx - 1) / bx,
                 ((uint32_t)length[1] + by - 1) / by, 1);
  } else {
    uint32_t bx = std::min(8u, (uint32_t)length[0]);
    uint32_t by = std::min(8u, (uint32_t)length[1]);
    uint32_t bz = max / (bx * by);
    block = uvec3(bx, by, bz);
    grid = uvec3(((uint32_t)length[0] + bx - 1) / bx,
                 ((uint32_t)length[1] + by - 1) / by,
                 ((uint32_t)(length_size > 2 ? length[2] : 1) + bz - 1) / bz);
  }

  d->dispatch(grid, block);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length, size_t length_size,
    const uint64_t* group_size, size_t group_size_size) {
  if (!func || !length || length_size == 0) return Error::InvalidArgument;
  auto* d = reinterpret_cast<KernelDispatcher*>(func);

  uint32_t bx, by, bz;
  if (length_size == 1) {
    bx = group_size && group_size_size > 0 ? (uint32_t)group_size[0] : (uint32_t)length[0];
    by = bz = 1;
  } else if (length_size == 2) {
    bx = group_size && group_size_size >= 2 ? (uint32_t)group_size[0] : 32;
    by = group_size && group_size_size >= 2 ? (uint32_t)group_size[1] : 32;
    bz = 1;
  } else {
    bx = group_size && group_size_size >= 3 ? (uint32_t)group_size[0] : 8;
    by = group_size && group_size_size >= 3 ? (uint32_t)group_size[1] : 8;
    bz = group_size && group_size_size >= 3 ? (uint32_t)group_size[2] : 8;
  }

  uvec3 block(bx, by, bz);
  uvec3 grid(((uint32_t)length[0] + bx - 1) / bx,
             (length_size > 1 ? ((uint32_t)length[1] + by - 1) / by : 1),
             (length_size > 2 ? ((uint32_t)length[2] + bz - 1) / bz : 1));
  d->dispatch(grid, block);
  return Error::Ok;
}

// --- Command block ---

void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func, void* user_data) {
  auto* wrapper =
      static_cast<std::function<void(AOTIMetalKernelFunctionHandle)>*>(user_data);
  if (wrapper) (*wrapper)(func);
}

AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data) {
  if (!func || !callback) return Error::InvalidArgument;
  // v1 used dispatch_sync_with_rethrow on the stream's serial GCD queue —
  // that's CPU-side serialization, NOT a GPU drain. v2 (single-threaded
  // AOTI) doesn't need either. The callback just encodes into the stream
  // (typically into the ICB), no GPU sync required. The earlier
  // getMetalStream()->wait() here was an over-translation: it triggered a
  // full flush+wait between every kernel callback, which (a) defeated
  // ICB's "encode many, drain at end" model and (b) caused stale ICB
  // re-execution that corrupted MPSGraph+ICB mixed models.
  callback(func, user_data);
  return Error::Ok;
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
