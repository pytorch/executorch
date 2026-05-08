/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI kernel-dispatch implementation built on MetalStream. Opaque AOTI
// handles point to file-local ShaderLibrary / KernelDispatcher structs.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/metal/core/MetalCommandRecorder.h>
#include <executorch/backends/metal/core/MetalKernelCache.h>
#include <executorch/backends/metal/core/MetalTypes.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

// Opaque handle types AOTI passes across the C ABI; the concrete
// definitions (ShaderLibrary / KernelDispatcher) are file-local.
struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;
struct AOTIMetalShaderLibraryOpaque;
using AOTIMetalShaderLibraryHandle = AOTIMetalShaderLibraryOpaque*;

typedef void (*aoti_torch_mps_command_block_callback_t)(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

using metal_v2::MetalKernel;
using metal_v2::MetalStream;
using metal_v2::uvec3;

// File-local opaque handle structs.

struct ShaderLibrary {
  std::string source;
  size_t source_hash = 0;  // computed once at create_shader_library.
};

struct KernelDispatcher {
  MetalKernel* kernel;  // owned by MetalKernelCache (process-wide).

  // Per-handle single-threaded contract: AOTI dispatches sequentially
  // against a given KernelDispatcher* (set_arg → dispatch). Concurrent
  // calls into the same handle from multiple threads are not safe;
  // `args` is mutated without synchronization.
  struct DeferredArg {
    // UNSET is the value-init default. If setBuffer/setScalarInt grow
    // args via vector::resize, untouched intermediate slots stay UNSET
    // and are skipped during dispatch.
    enum Kind { UNSET = 0, BUFFER, SCALAR } kind = UNSET;
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

  void dispatch(uvec3 grid, uvec3 block) {
    auto* stream = getMetalStream();

    for (auto& a : args) {
      if (a.kind == DeferredArg::BUFFER && a.ptr) {
        stream->allocator().registerExternalBuffer(a.ptr, a.size);
      }
    }

    // BUFFER args use setInOut because AOTI's set_arg_tensor ABI doesn't
    // distinguish input from output. Tracking each as both read+write
    // forces a barrier on the next dispatch that reads the same buffer
    // — which is required to avoid a RAW hazard on chained dispatches.
    auto d = stream->recorder().beginDispatch(kernel);
    for (size_t i = 0; i < args.size(); ++i) {
      auto& a = args[i];
      const uint32_t slot = static_cast<uint32_t>(i);
      if (a.kind == DeferredArg::UNSET) continue;
      if (a.kind == DeferredArg::BUFFER) {
        d.setInOut(slot, a.ptr, a.size);
      } else {
        d.setBytes(slot, a.scalar, sizeof(int64_t));
      }
    }
    d.run(grid, block);

    clear();
  }
};

// shared_ptr (not unique_ptr) so get_kernel_function can extend the
// library's lifetime past a concurrent delete_shader_library call.
static std::mutex storage_mutex;
static std::unordered_map<ShaderLibrary*, std::shared_ptr<ShaderLibrary>> library_storage;
static std::unordered_map<KernelDispatcher*, std::unique_ptr<KernelDispatcher>> function_storage;

// Grid/block helpers.

// Returns true if every length fits in uint32_t (the grid/block widths
// Metal requires). Logs and returns false on overflow.
static bool checkLengthsFitU32(const uint64_t* length, size_t n,
                               const char* shim_name) {
  for (size_t i = 0; i < n; ++i) {
    if (length[i] > UINT32_MAX) {
      ET_LOG(Error, "%s: length[%zu]=%llu exceeds UINT32_MAX",
          shim_name, i, (unsigned long long)length[i]);
      return false;
    }
  }
  return true;
}

static uvec3 computeBlock1D(MetalKernel* k, uint64_t length) {
  uint32_t max = k->maxThreadsPerThreadgroup().x;
  return uvec3(std::min(max, (uint32_t)length), 1, 1);
}

static uvec3 computeGrid1D(uint64_t length, uint32_t blockX) {
  return uvec3(((uint32_t)length + blockX - 1) / blockX, 1, 1);
}

extern "C" {

// Shader library.

AOTITorchError aoti_torch_mps_create_shader_library(
    const char* source,
    AOTIMetalShaderLibraryHandle* out) {
  if (!source || !out) return Error::InvalidArgument;

  size_t src_len = std::strlen(source);
  ET_LOG(Debug,
      "[shader-src] create_shader_library len=%zu source:\n----\n%.1500s%s\n----",
      src_len, source, src_len > 1500 ? "\n... [truncated]" : "");

  auto lib = std::make_shared<ShaderLibrary>();
  lib->source = source;
  lib->source_hash = std::hash<std::string>{}(lib->source);
  auto* raw = lib.get();
  {
    std::lock_guard<std::mutex> lk(storage_mutex);
    library_storage[raw] = std::move(lib);
  }
  *out = reinterpret_cast<AOTIMetalShaderLibraryHandle>(raw);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle handle) {
  if (!handle) return Error::InvalidArgument;
  auto* lib = reinterpret_cast<ShaderLibrary*>(handle);
  std::lock_guard<std::mutex> lk(storage_mutex);
  library_storage.erase(lib);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle lib_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* out) {
  if (!lib_handle || !kernel_name || !out) return Error::InvalidArgument;

  // Copy the shared_ptr under the lock so the library outlives a
  // concurrent delete_shader_library call.
  std::shared_ptr<ShaderLibrary> lib;
  {
    std::lock_guard<std::mutex> lk(storage_mutex);
    auto it = library_storage.find(reinterpret_cast<ShaderLibrary*>(lib_handle));
    if (it == library_storage.end()) {
      ET_LOG(Error, "get_kernel_function: unknown library handle %p", lib_handle);
      return Error::InvalidArgument;
    }
    lib = it->second;
  }

  // Cache key derived from the library SOURCE so a recycled library
  // address (after delete + new alloc) cannot return the wrong kernel.
  // Including source.size() makes hash collisions effectively impossible
  // (two distinct sources would have to collide on both hash and length).
  std::string cache_key = std::to_string(lib->source_hash) + ":"
      + std::to_string(lib->source.size()) + "::" + kernel_name;
  MetalKernel* kernel = metal_v2::MetalKernelCache::shared().findOrInsert(
      cache_key, [&]() {
        return getMetalStream()->compiler()->compile(
            lib->source.c_str(), kernel_name);
      });
  if (!kernel) {
    ET_LOG(Error, "Failed to compile kernel '%s'", kernel_name);
    return Error::Internal;
  }

  auto disp = std::make_unique<KernelDispatcher>();
  disp->kernel = kernel;
  auto* raw = disp.get();
  {
    std::lock_guard<std::mutex> lk(storage_mutex);
    function_storage[raw] = std::move(disp);
  }
  *out = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw);
  return Error::Ok;
}

// Encoding / args / dispatch.

AOTITorchError aoti_torch_mps_start_encoding(AOTIMetalKernelFunctionHandle func) {
  if (!func) return Error::InvalidArgument;
  reinterpret_cast<KernelDispatcher*>(func)->clear();
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func, unsigned idx, AOTITensorHandle tensor) {
  if (!func || !tensor) return Error::InvalidArgument;
  if (idx >= metal_v2::MetalCommandRecorder::kMaxBuffersPerDispatch) {
    ET_LOG(Error, "set_arg_tensor: slot %u exceeds max %zu",
        idx, metal_v2::MetalCommandRecorder::kMaxBuffersPerDispatch);
    return Error::InvalidArgument;
  }
  auto* t = reinterpret_cast<Tensor*>(tensor);
  reinterpret_cast<KernelDispatcher*>(func)->setBuffer(
      idx, t->data_ptr(), t->nbytes());
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func, unsigned idx, int64_t val) {
  if (!func) return Error::InvalidArgument;
  if (idx >= metal_v2::MetalCommandRecorder::kMaxBuffersPerDispatch) {
    ET_LOG(Error, "set_arg_int: slot %u exceeds max %zu",
        idx, metal_v2::MetalCommandRecorder::kMaxBuffersPerDispatch);
    return Error::InvalidArgument;
  }
  reinterpret_cast<KernelDispatcher*>(func)->setScalarInt(idx, val);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func, uint64_t length) {
  if (!func) return Error::InvalidArgument;
  if (length == 0) return Error::Ok;  // No-op for empty grids.
  if (!checkLengthsFitU32(&length, 1, "dispatch_single")) {
    return Error::InvalidArgument;
  }
  auto* d = reinterpret_cast<KernelDispatcher*>(func);
  uvec3 block = computeBlock1D(d->kernel, length);
  d->dispatch(computeGrid1D(length, block.x), block);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func, uint64_t length, uint64_t group_size) {
  if (!func) return Error::InvalidArgument;
  if (length == 0) return Error::Ok;
  if (!checkLengthsFitU32(&length, 1, "dispatch_single_with_group_size")) {
    return Error::InvalidArgument;
  }
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
  for (size_t i = 0; i < length_size; ++i) {
    if (length[i] == 0) return Error::Ok;
  }
  if (!checkLengthsFitU32(length, length_size, "dispatch_array")) {
    return Error::InvalidArgument;
  }
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
                 ((uint32_t)length[2] + bz - 1) / bz);
  }

  d->dispatch(grid, block);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length, size_t length_size,
    const uint64_t* group_size, size_t group_size_size) {
  if (!func || !length || length_size == 0) return Error::InvalidArgument;
  for (size_t i = 0; i < length_size; ++i) {
    if (length[i] == 0) return Error::Ok;
  }
  if (!checkLengthsFitU32(length, length_size, "dispatch_array_with_group_size")) {
    return Error::InvalidArgument;
  }
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

  // Reject group sizes that exceed the kernel's max threads/threadgroup
  // — Metal validation would otherwise fault deep inside the encoder.
  uint64_t threads = (uint64_t)bx * by * bz;
  uint32_t max_threads = d->kernel->maxThreadsPerThreadgroup().x;
  if (threads == 0 || threads > max_threads) {
    ET_LOG(Error,
        "dispatch_array_with_group_size: group %ux%ux%u (=%llu threads) "
        "exceeds kernel max %u",
        bx, by, bz, (unsigned long long)threads, max_threads);
    return Error::InvalidArgument;
  }

  uvec3 block(bx, by, bz);
  uvec3 grid(((uint32_t)length[0] + bx - 1) / bx,
             (length_size > 1 ? ((uint32_t)length[1] + by - 1) / by : 1),
             (length_size > 2 ? ((uint32_t)length[2] + bz - 1) / bz : 1));
  d->dispatch(grid, block);
  return Error::Ok;
}

// Command block.

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
  // The callback encodes into the active stream; no GPU sync required.
  callback(func, user_data);
  return Error::Ok;
}

}  // extern "C"

}  // namespace metal
}  // namespace backends
}  // namespace executorch
