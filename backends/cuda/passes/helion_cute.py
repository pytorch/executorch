# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile a Helion CuTe kernel launch into a C function AOTInductor links.

The CUDA backend cannot run Helion's CuTe launcher at inference: it needs Python,
Helion and the CuTe DSL. This module compiles one launch ahead of time instead.
The CuTe DSL exports the compiled launch as an object file with a plain C entry
point (``export_to_c``), and a generated C shim adapts that entry point to the
AOTInductor custom-op ABI. Both go into a static library the AOTInductor
``.so`` links, so the kernel ends up inside the ``.pte`` like the Triton kernels
AOTInductor compiles. The launch is registered as a custom op whose calls
AOTInductor lowers to the shim.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast, NamedTuple

import torch
from torch._inductor.cpp_builder import get_cpp_compiler
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.multiprocessing.reductions import StorageWeakRef

_OP_NAMESPACE = "helion_cute"
_DSL_REQUIREMENT = (
    "Helion with CuTe backend support and nvidia-cutlass-dsl 4.7 or newer"
)
_MIN_DSL_VERSION = (4, 7)
_TVM_FFI_OPTION = "--enable-tvm-ffi"
# The kwargs a Helion CuTe launch site may pass; the export reproduces both.
_LAUNCH_KWARGS = frozenset({"block", "cute_compile_options"})
_DEFAULT_BLOCK = (256, 1, 1)

_SHIM_TEMPLATE = """\
#include <cstdint>
#include <cstdio>
#include <mutex>

#include <cuda_runtime.h>

// The exported header loads the kernel module through this macro; record the
// first failure instead of only printing it, so the call can return an error.
static cudaError_t module_load_status = cudaSuccess;
#define CUTE_DSL_CUDA_ERROR_CHECK(err)                                \\
  {{                                                                  \\
    const cudaError_t status_ = (err);                                \\
    if (status_ != cudaSuccess && module_load_status == cudaSuccess) {{ \\
      module_load_status = status_;                                   \\
    }}                                                                \\
  }}
#include "{prefix}.h"

struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;
using AOTITorchError = int32_t;
extern "C" AOTITorchError aoti_torch_get_data_ptr(AtenTensorHandle, void**);
extern "C" AOTITorchError aoti_torch_get_dim(AtenTensorHandle, int64_t*);
extern "C" AOTITorchError aoti_torch_get_sizes(AtenTensorHandle, int64_t**);
extern "C" AOTITorchError aoti_torch_get_strides(AtenTensorHandle, int64_t**);
extern "C" AOTITorchError aoti_torch_get_device_index(AtenTensorHandle, int32_t*);
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t, void**);

namespace {{
constexpr int kCount = {count};
constexpr int64_t kRanks[kCount] = {{{ranks}}};
{layout_arrays}
const int64_t* const kSizes[kCount] = {{{sizes}}};
const int64_t* const kStrides[kCount] = {{{strides}}};
constexpr uintptr_t kAlignments[kCount] = {{{alignments}}};

bool matches_export(AtenTensorHandle tensor, int i, void** pointer) {{
  int64_t rank = 0;
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  if (aoti_torch_get_data_ptr(tensor, pointer) != 0 ||
      aoti_torch_get_dim(tensor, &rank) != 0 ||
      aoti_torch_get_sizes(tensor, &sizes) != 0 ||
      aoti_torch_get_strides(tensor, &strides) != 0 || rank != kRanks[i]) {{
    return false;
  }}
  for (int64_t d = 0; d < rank; ++d) {{
    // A size-1 dimension's stride never affects an address, and Inductor's
    // exact-stride matching ignores it, so the caller's stride there is
    // free to differ from the exported one.
    if (sizes[d] != kSizes[i][d] ||
        (sizes[d] != 1 && strides[d] != kStrides[i][d])) {{
      return false;
    }}
  }}
  return reinterpret_cast<uintptr_t>(*pointer) % kAlignments[i] == 0;
}}
}} // namespace

extern "C" AOTITorchError {shim}({parameters}, AtenTensorHandle* /*unused*/) {{
  static {prefix}_Kernel_Module_t module;
  static std::once_flag loaded;
  std::call_once(loaded, [] {{ {prefix}_Kernel_Module_Load(&module); }});
  if (module_load_status != cudaSuccess) {{
    fprintf(stderr, "{prefix}: cannot load the kernel module: %s\\n",
            cudaGetErrorString(module_load_status));
    return 1;
  }}
  const AtenTensorHandle tensors[kCount] = {{{handles}}};
  void* pointers[kCount];
  for (int i = 0; i < kCount; ++i) {{
    if (!matches_export(tensors[i], i, &pointers[i])) {{
      fprintf(stderr,
              "{prefix}: argument %d does not have the layout or alignment "
              "the kernel was compiled for\\n", i);
      return 1;
    }}
  }}
  int32_t device = 0;
  void* stream = nullptr;
  if (aoti_torch_get_device_index(tensors[0], &device) != 0 ||
      aoti_torch_get_current_cuda_stream(device, &stream) != 0) {{
    fprintf(stderr, "{prefix}: cannot get the current CUDA stream\\n");
    return 1;
  }}
  const int32_t status = cute_dsl_{prefix}_wrapper(
      &module, {pointer_args}, {grid}, static_cast<cudaStream_t>(stream));
  if (status != 0) {{
    fprintf(stderr, "{prefix}: the CuTe launch failed with %d\\n", status);
  }}
  return status;
}}
"""

# The exported kernel object calls CUDA through `_cuda*` / `_cu*` symbols.
# Those are one-line forwards to the public CUDA runtime and driver APIs, so
# this file implements them directly instead of linking the CuTe DSL wheel's
# proprietary static runtime library. The forwards are built once into a
# single archive (`_RUNTIME_LIBRARY`) that every program links once, however
# many CuTe launches it contains. The `_cu*` driver entry points are resolved
# through the runtime, so no direct link against libcuda is needed either.
_RUNTIME_LIBRARY = "helion_cute_runtime"
_RUNTIME_TEMPLATE = """\
#include <cuda.h>
#include <dlfcn.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

extern "C" {

cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return cudaDeviceGetAttribute(value, attr, device);
}
cudaError_t _cudaDeviceSynchronize() { return cudaDeviceSynchronize(); }
cudaError_t _cudaEventCreate(cudaEvent_t* event) {
  return cudaEventCreate(event);
}
cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  return cudaEventCreateWithFlags(event, flags);
}
cudaError_t _cudaEventDestroy(cudaEvent_t event) {
  return cudaEventDestroy(event);
}
cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  return cudaEventElapsedTime(ms, start, end);
}
cudaError_t _cudaEventQuery(cudaEvent_t event) { return cudaEventQuery(event); }
cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return cudaEventRecord(event, stream);
}
cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                      unsigned int flags) {
  return cudaEventRecordWithFlags(event, stream, flags);
}
cudaError_t _cudaEventSynchronize(cudaEvent_t event) {
  return cudaEventSynchronize(event);
}
cudaError_t _cudaFree(void* ptr) { return cudaFree(ptr); }
cudaError_t _cudaFreeAsync(void* ptr, cudaStream_t stream) {
  return cudaFreeAsync(ptr, stream);
}
cudaError_t _cudaFreeHost(void* ptr) { return cudaFreeHost(ptr); }
cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr,
                                  int value) {
  return cudaFuncSetAttribute(func, attr, value);
}
cudaError_t _cudaGetDevice(int* device) { return cudaGetDevice(device); }
cudaError_t _cudaGetDeviceCount(int* count) {
  return cudaGetDeviceCount(count);
}
cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}
cudaError_t _cudaGetLastError() { return cudaGetLastError(); }
cudaError_t _cudaGraphDestroy(cudaGraph_t graph) {
  return cudaGraphDestroy(graph);
}
cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t exec) {
  return cudaGraphExecDestroy(exec);
}
cudaError_t _cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
  return cudaGraphLaunch(exec, stream);
}
cudaError_t _cudaLaunchKernel(const void* func, dim3 grid, dim3 block,
                              void** args, size_t shared_mem,
                              cudaStream_t stream) {
  return cudaLaunchKernel(func, grid, block, args, shared_mem, stream);
}
cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t* config,
                                const void* func, void** args) {
  return cudaLaunchKernelExC(config, func, args);
}
cudaError_t _cudaLibraryGetGlobal(void** ptr, size_t* bytes,
                                  cudaLibrary_t library, const char* name) {
  return cudaLibraryGetGlobal(ptr, bytes, library, name);
}
cudaError_t _cudaLibraryGetKernel(cudaKernel_t* kernel, cudaLibrary_t library,
                                  const char* name) {
  return cudaLibraryGetKernel(kernel, library, name);
}
cudaError_t _cudaLibraryGetKernelCount(unsigned int* count,
                                       cudaLibrary_t library) {
  return cudaLibraryGetKernelCount(count, library);
}
cudaError_t _cudaLibraryLoadData(cudaLibrary_t* library, const void* image,
                                 cudaJitOption* jit_options,
                                 void** jit_option_values,
                                 unsigned int num_jit_options,
                                 cudaLibraryOption* library_options,
                                 void** library_option_values,
                                 unsigned int num_library_options) {
  return cudaLibraryLoadData(library, image, jit_options, jit_option_values,
                             num_jit_options, library_options,
                             library_option_values, num_library_options);
}
cudaError_t _cudaLibraryUnload(cudaLibrary_t library) {
  return cudaLibraryUnload(library);
}
cudaError_t _cudaMalloc(void** ptr, size_t size) {
  return cudaMalloc(ptr, size);
}
cudaError_t _cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return cudaMallocAsync(ptr, size, stream);
}
cudaError_t _cudaMallocHost(void** ptr, size_t size) {
  return cudaMallocHost(ptr, size);
}
cudaError_t _cudaMallocManaged(void** ptr, size_t size,
                               unsigned int flags) {
  return cudaMallocManaged(ptr, size, flags);
}
cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count,
                        cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, count, kind);
}
cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}
cudaError_t _cudaMemset(void* ptr, int value, size_t count) {
  return cudaMemset(ptr, value, count);
}
cudaError_t _cudaMemsetAsync(void* ptr, int value, size_t count,
                             cudaStream_t stream) {
  return cudaMemsetAsync(ptr, value, count, stream);
}
cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* num_blocks, const void* func, int block_size, size_t dynamic_smem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks, func,
                                                       block_size, dynamic_smem);
}
cudaError_t _cudaProfilerStart() { return cudaProfilerStart(); }
cudaError_t _cudaProfilerStop() { return cudaProfilerStop(); }
cudaError_t _cudaSetDevice(int device) { return cudaSetDevice(device); }

cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* node, cudaGraph_t graph,
                                    const cudaGraphNode_t* dependencies,
                                    size_t num_dependencies,
                                    const cudaKernelNodeParams* params) {
  return cudaGraphAddKernelNode(node, graph, dependencies, num_dependencies,
                                params);
}
cudaError_t _cudaGraphCreate(cudaGraph_t* graph, unsigned int flags) {
  return cudaGraphCreate(graph, flags);
}
cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                  unsigned long long flags) {
  return cudaGraphInstantiate(exec, graph, flags);
}
cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t node,
                                             cudaKernelNodeAttrID attr,
                                             cudaKernelNodeAttrValue* value) {
  return cudaGraphKernelNodeGetAttribute(node, attr, value);
}
cudaError_t _cudaGraphKernelNodeSetAttribute(
    cudaGraphNode_t node, cudaKernelNodeAttrID attr,
    const cudaKernelNodeAttrValue* value) {
  return cudaGraphKernelNodeSetAttribute(node, attr, value);
}
cudaError_t _cudaKernelSetAttributeForDevice(const void* kernel, int attr,
                                             const void* value, int device) {
  // The attribute types were renamed between CUDA releases; the ABI is
  // (pointer, enum, pointer, int) in every one, so resolve the runtime
  // entry point once and forward opaquely.
  using Fn = cudaError_t (*)(const void*, int, const void*, int);
  static const auto fn = [] {
    // The CUDA runtime may only be a local dependency of this .so, so
    // search its own handles rather than the global symbol scope.
    for (const char* name :
         {"libcudart.so.13", "libcudart.so.12", "libcudart.so"}) {
      void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        if (void* symbol =
                dlsym(handle, "cudaKernelSetAttributeForDevice")) {
          return reinterpret_cast<Fn>(symbol);
        }
      }
    }
    return reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "cudaKernelSetAttributeForDevice"));
  }();
  if (fn == nullptr) {
    return cudaErrorInvalidValue;
  }
  return fn(kernel, attr, value, device);
}
cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* params) {
  return cudaMemcpy3D(params);
}
cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* params,
                               cudaStream_t stream) {
  return cudaMemcpy3DAsync(params, stream);
}
cudaError_t _cudaStreamAddCallback(cudaStream_t stream,
                                   cudaStreamCallback_t callback,
                                   void* user_data, unsigned int flags) {
  return cudaStreamAddCallback(stream, callback, user_data, flags);
}
cudaError_t _cudaStreamBeginCapture(cudaStream_t stream,
                                    cudaStreamCaptureMode mode) {
  return cudaStreamBeginCapture(stream, mode);
}
cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
  return cudaStreamEndCapture(stream, graph);
}
cudaError_t _cudaStreamCreate(cudaStream_t* stream) {
  return cudaStreamCreate(stream);
}
cudaError_t _cudaStreamDestroy(cudaStream_t stream) {
  return cudaStreamDestroy(stream);
}
cudaError_t _cudaStreamSynchronize(cudaStream_t stream) {
  return cudaStreamSynchronize(stream);
}
cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                 unsigned int flags) {
  return cudaStreamWaitEvent(stream, event, flags);
}

} // extern "C"

namespace {
void* driver_symbol(const char* name) {
  static std::mutex mutex;
  static std::unordered_map<const char*, void*> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }
  void* fn = nullptr;
  // 12080 is the CUDA 12.8 driver version the CuTe DSL 4.7 runtime targets.
  cudaGetDriverEntryPointByVersion(name, &fn, 12080, cudaEnableDefault,
                                   nullptr);
  cache[name] = fn;
  return fn;
}

template <typename Fn, typename... Args>
CUresult call_driver(const char* name, Args... args) {
  void* fn = driver_symbol(name);
  if (fn == nullptr) {
    return CUDA_ERROR_UNKNOWN;
  }
  return reinterpret_cast<Fn>(fn)(args...);
}
} // namespace

extern "C" {

CUresult _cuDeviceGet(CUdevice* device, int ordinal) {
  using Fn = CUresult (*)(CUdevice*, int);
  return call_driver<Fn>("cuDeviceGet", device, ordinal);
}
CUresult _cuEventCreate(CUevent* event, unsigned int flags) {
  using Fn = CUresult (*)(CUevent*, unsigned int);
  return call_driver<Fn>("cuEventCreate", event, flags);
}
CUresult _cuEventDestroy_v2(CUevent event) {
  using Fn = CUresult (*)(CUevent);
  return call_driver<Fn>("cuEventDestroy_v2", event);
}
CUresult _cuEventElapsedTime_v2(float* ms, CUevent start, CUevent end) {
  using Fn = CUresult (*)(float*, CUevent, CUevent);
  return call_driver<Fn>("cuEventElapsedTime_v2", ms, start, end);
}
CUresult _cuEventQuery(CUevent event) {
  using Fn = CUresult (*)(CUevent);
  return call_driver<Fn>("cuEventQuery", event);
}
CUresult _cuEventRecord(CUevent event, CUstream stream) {
  using Fn = CUresult (*)(CUevent, CUstream);
  return call_driver<Fn>("cuEventRecord", event, stream);
}
CUresult _cuEventRecordWithFlags(CUevent event, CUstream stream,
                                 unsigned int flags) {
  using Fn = CUresult (*)(CUevent, CUstream, unsigned int);
  return call_driver<Fn>("cuEventRecordWithFlags", event, stream, flags);
}
CUresult _cuEventSynchronize(CUevent event) {
  using Fn = CUresult (*)(CUevent);
  return call_driver<Fn>("cuEventSynchronize", event);
}
CUresult _cuKernelGetAttribute(int* value, CUfunction_attribute attr,
                               CUfunction function, CUdevice device) {
  using Fn = CUresult (*)(int*, CUfunction_attribute, CUfunction, CUdevice);
  return call_driver<Fn>("cuKernelGetAttribute", value, attr, function, device);
}
CUresult _cuLibraryGetGlobal(CUdeviceptr* ptr, size_t* bytes,
                             CUlibrary library, const char* name) {
  using Fn = CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*);
  return call_driver<Fn>("cuLibraryGetGlobal", ptr, bytes, library, name);
}
CUresult _cuLibraryGetKernelCount(unsigned int* count, CUlibrary library) {
  using Fn = CUresult (*)(unsigned int*, CUlibrary);
  return call_driver<Fn>("cuLibraryGetKernelCount", count, library);
}
CUresult _cuLibraryUnload(CUlibrary library) {
  using Fn = CUresult (*)(CUlibrary);
  return call_driver<Fn>("cuLibraryUnload", library);
}
CUresult _cuTensorMapEncodeTiled(
    CUtensorMap* tensor_map, CUtensorMapDataType data_type,
    cuuint32_t tensor_rank, void* global_address,
    const cuuint64_t* global_dim, const cuuint64_t* global_strides,
    const cuuint32_t* box_dim, const cuuint32_t* element_strides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2_promotion, CUtensorMapFloatOOBfill oob_fill) {
  using Fn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                          const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*,
                          CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  return call_driver<Fn>("cuTensorMapEncodeTiled", tensor_map, data_type,
                         tensor_rank, global_address, global_dim,
                         global_strides, box_dim, element_strides, interleave,
                         swizzle, l2_promotion, oob_fill);
}

} // extern "C"
"""


class _ExportedLaunch(NamedTuple):
    op: Any
    shim_declaration: str
    library: str
    library_directory: Path


class _ProgramLaunches(threading.local):
    def __init__(self) -> None:
        self.launches: dict[str, _ExportedLaunch] = {}
        self.last_had_launches = False


class _ExportState:
    """The launches exported in this process, and those of each thread's program.

    Shim declarations are (re)written on the compiling thread by
    ``link_exported_launches``, because Inductor's config is thread-local.
    The supported fallback kernels and the archive *link* are per program:
    ``AotiBackend.preprocess`` runs
    ``LowerHelionPass`` and then, on the same thread, enters
    ``link_exported_launches`` for the compile, so a thread's program
    launches are exactly the archives its next compile on that thread links.
    Calling ``LowerHelionPass`` outside that flow leaves the thread's
    launches recorded until the next pass run (``begin_program``) or link.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.library: torch.library.Library | None = None
        self.exported: dict[str, _ExportedLaunch] = {}
        self.prefix_locks: dict[str, threading.Lock] = {}
        self.library_directories: set[Path] = set()
        self.cleanup_registered = False
        self.per_thread = _ProgramLaunches()

    def program_launches(self) -> dict[str, _ExportedLaunch]:
        return self.per_thread.launches

    def prefix_lock(self, prefix: str) -> threading.Lock:
        with self.lock:
            lock = self.prefix_locks.get(prefix)
            if lock is None:
                lock = threading.Lock()
                self.prefix_locks[prefix] = lock
            return lock

    def track_library_directory(self, directory: Path) -> None:
        self.library_directories.add(directory)
        if not self.cleanup_registered:
            self.cleanup_registered = True
            atexit.register(self._remove_library_directories)

    def _remove_library_directories(self) -> None:
        for directory in self.library_directories:
            shutil.rmtree(directory, ignore_errors=True)


_STATE = _ExportState()


class _CapturedLaunch(NamedTuple):
    cute_kernel: Any
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    compile_options: str | None
    tensors: tuple[torch.Tensor, ...]


def _xyz(dims: Sequence[object]) -> tuple[int, int, int]:
    values = []
    for value in dims:
        if not isinstance(value, int):
            raise NotImplementedError(
                f"Helion CuTe lowering needs static launch dimensions, got {value!r}"
            )
        values.append(value)
    x, y, z = (values + [1, 1, 1])[:3]
    return x, y, z


def _capture_launch(
    entrypoint: Callable[..., Any],
    launcher_parameter: str,
    arguments: tuple[torch.Tensor, ...],
) -> _CapturedLaunch:
    launches: list[
        tuple[Any, Sequence[object], tuple[object, ...], dict[str, object]]
    ] = []

    def capture(
        cute_kernel: Any, grid: Sequence[object], *args: object, **kwargs: object
    ) -> None:
        launches.append((cute_kernel, grid, args, kwargs))

    with torch.no_grad():
        entrypoint(*arguments, **{launcher_parameter: capture})
    if len(launches) != 1:
        raise NotImplementedError("Helion CuTe lowering requires a single launch")
    cute_kernel, grid, args, kwargs = launches[0]
    unsupported = set(kwargs) - _LAUNCH_KWARGS
    if unsupported:
        raise NotImplementedError(
            f"Helion CuTe lowering does not support launch options {sorted(unsupported)}"
        )
    if not args or not all(isinstance(a, torch.Tensor) for a in args):
        raise NotImplementedError(
            "Helion CuTe lowering supports launches with tensor arguments only"
        )
    options = kwargs.get("cute_compile_options")
    if options is not None and not isinstance(options, str):
        raise NotImplementedError(f"Unexpected CuTe compile options {options!r}")
    block = kwargs.get("block", _DEFAULT_BLOCK)
    if not isinstance(block, (tuple, list)):
        raise NotImplementedError(f"Unexpected CuTe block {block!r}")
    grid_xyz = _xyz(grid)
    if min(grid_xyz) <= 0:
        raise NotImplementedError(f"Helion CuTe lowering found an empty grid {grid!r}")
    return _CapturedLaunch(
        cute_kernel,
        grid_xyz,
        _xyz(block),
        options,
        tuple(cast(torch.Tensor, a) for a in args),
    )


def _example_like(fake: torch.Tensor, aligned: bool) -> torch.Tensor:
    """A real tensor with ``fake``'s layout to compile the launch for.

    Helion derives each pointer's assumed alignment from the example tensors. A
    tensor that may be a view at run time, with a storage offset the exported
    metadata does not show, gets an example one element into its storage, so
    the kernel assumes only element alignment for it.
    """
    # A root parameter or buffer that is itself a view keeps its storage
    # offset in the .pte (AOTI does not clone parameters), so even an
    # "aligned" argument must be compiled for its offset's alignment
    # class: only the offset modulo the 16-byte assumption matters to
    # Helion, and reproducing a huge real offset would just waste export
    # memory. Other aligned arguments start at their allocation.
    if aligned:
        elements_per_16_bytes = max(1, 16 // fake.element_size())
        offset = fake.storage_offset() % elements_per_16_bytes
    else:
        offset = 1
    extent = 1 + sum((n - 1) * s for n, s in zip(fake.shape, fake.stride()))
    base = torch.empty(offset + extent, dtype=fake.dtype, device=fake.device)
    return base.as_strided(fake.shape, fake.stride(), offset)


_RUNTIME_ARCHIVE_LOCK = threading.Lock()
_RUNTIME_ARCHIVE_DIRECTORY: Path | None = None


def _runtime_archive_directory() -> Path:
    """Build this process's one CuTe dialect runtime archive, once.

    The archive holds only the `_cuda*` / `_cu*` forwards in
    `_RUNTIME_TEMPLATE`. Every program links it once, however many launches
    it contains, so a `.pte` never carries more than one copy.
    """
    global _RUNTIME_ARCHIVE_DIRECTORY
    with _RUNTIME_ARCHIVE_LOCK:
        if _RUNTIME_ARCHIVE_DIRECTORY is not None:
            return _RUNTIME_ARCHIVE_DIRECTORY
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            raise RuntimeError(
                "Helion CuTe lowering needs a CUDA toolkit (CUDA_HOME)"
            )
        digest = hashlib.sha256(_RUNTIME_TEMPLATE.encode()).hexdigest()[:16]
        root = Path(cache_dir()) / "executorch_helion_cute"
        directory = root / f"runtime_{digest}"
        archive = directory / f"lib{_RUNTIME_LIBRARY}.a"
        if not archive.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                workdir = Path(temporary)
                (workdir / "dialect_runtime.cpp").write_text(_RUNTIME_TEMPLATE)
                try:
                    subprocess.run(
                        [
                            get_cpp_compiler(),
                            "-c",
                            "-fPIC",
                            "-O2",
                            "-std=c++17",
                            f"-I{CUDA_HOME}/include",
                            str(workdir / "dialect_runtime.cpp"),
                            "-o",
                            str(workdir / "dialect_runtime.o"),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            shutil.which("ar") or "ar",
                            "rcs",
                            str(workdir / f"lib{_RUNTIME_LIBRARY}.a"),
                            str(workdir / "dialect_runtime.o"),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        "Helion CuTe lowering failed to build the dialect "
                        f"runtime archive: {error.stderr}"
                    ) from error
                except OSError as error:
                    raise RuntimeError(
                        "Helion CuTe lowering cannot run the C++ compiler "
                        f"or ar: {error}"
                    ) from error
                shutil.copyfile(
                    workdir / f"lib{_RUNTIME_LIBRARY}.a", archive
                )
        _RUNTIME_ARCHIVE_DIRECTORY = directory
        return directory


@functools.cache
def _cute_launcher_modules() -> tuple[Any, Any, Any, Any]:
    """Helion's private CuTe launcher modules, checked against what we call.

    Lowering builds on private Helion launcher functions and the CuTe DSL, so
    probe them up front: a Helion or DSL that renamed one must fail with the
    requirement error instead of a bare AttributeError or ImportError.
    """
    try:
        dsl_version: str | None = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        dsl_version = None
    if dsl_version is not None:
        parsed = tuple(
            int(part) for part in dsl_version.split(".")[:2] if part.isdigit()
        )
        if parsed and parsed < _MIN_DSL_VERSION:
            raise RuntimeError(
                f"Helion CuTe lowering needs {_DSL_REQUIREMENT}; "
                f"found nvidia-cutlass-dsl {dsl_version}"
            )
    try:
        launcher = importlib.import_module("helion.runtime.cute.launcher")
        arg_signature = importlib.import_module("helion.runtime.cute.arg_signature")
        sass_check = importlib.import_module("helion.runtime.cute.sass_check")
        cute = importlib.import_module("cutlass.cute")
    except ImportError as error:
        raise RuntimeError(f"Helion CuTe lowering needs {_DSL_REQUIREMENT}") from error
    required = (
        (launcher, "_cute_stage_under_aligned_operands"),
        (launcher, "_build_cute_schema_and_args"),
        (launcher, "_ensure_cute_dsl_arch_env"),
        (launcher, "_create_cute_wrapper"),
        (launcher, "_cute_num_sm_from_arch_args"),
        (launcher, "_cute_current_stream"),
        (arg_signature, "schema_pointer_alignment"),
        (sass_check, "verify_compiled"),
        (cute, "compile"),
    )
    missing = [
        f"{module.__name__}.{name}"
        for module, name in required
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"Helion CuTe lowering needs {_DSL_REQUIREMENT}; "
            f"this install lacks {', '.join(missing)}"
        )
    return launcher, arg_signature, sass_check, cute


def _check_error_check_macro(header: str) -> None:
    """Reject a header that would replace the shim's load-error override.

    The shim defines ``CUTE_DSL_CUDA_ERROR_CHECK`` before including the
    exported header so module-load failures are recorded. A header that
    defines the macro again outside an ``#ifndef`` guard silently replaces
    that override (a warning the captured compiler output never surfaces).
    """
    lines = header.splitlines()
    uses = 0
    for index, line in enumerate(lines):
        if re.match(r"\s*#\s*define\s+CUTE_DSL_CUDA_ERROR_CHECK\b", line):
            previous = [text.strip() for text in lines[:index] if text.strip()]
            if previous and previous[-1].startswith(
                "#ifndef CUTE_DSL_CUDA_ERROR_CHECK"
            ):
                continue
            raise NotImplementedError(
                "Helion CuTe lowering found an exported header that redefines "
                "CUTE_DSL_CUDA_ERROR_CHECK without an #ifndef guard"
            )
        if "CUTE_DSL_CUDA_ERROR_CHECK(" in line:
            uses += 1
    if uses == 0:
        raise NotImplementedError(
            "Helion CuTe lowering found an exported header that never uses "
            "CUTE_DSL_CUDA_ERROR_CHECK, so module load errors would not be "
            "reported"
        )


def _compile_and_export(
    captured: _CapturedLaunch, prefix: str, workdir: Path
) -> tuple[Any, list[int]]:
    """Compile the launch with Helion's own wrapper and export it to C."""
    launcher, arg_signature, sass_check, cute = _cute_launcher_modules()
    tensors = captured.tensors
    if launcher._cute_stage_under_aligned_operands(captured.cute_kernel, tensors):
        raise NotImplementedError(
            "Helion CuTe lowering does not support launches that copy "
            "under-aligned operands"
        )
    launch = launcher._build_cute_schema_and_args(
        captured.cute_kernel, tensors, captured.grid
    )
    schema = tuple(launch.schema)
    if launch.owned_tensors or len(schema) != len(tensors):
        raise NotImplementedError(
            "Helion CuTe lowering does not support launches with workspace tensors"
        )
    alignments = []
    for entry in schema:
        if entry[0] != "tensor" or len(entry) not in (5, 6):
            raise NotImplementedError(
                f"Helion CuTe lowering supports static tensor arguments only, got {entry[0]!r}"
            )
        alignments.append(arg_signature.schema_pointer_alignment(entry))
    launch_args = tuple(launch.launch_args)
    if (
        len(launch_args) != len(tensors) + 3
        or tuple(int(a) for a in launch_args[-3:]) != captured.grid
    ):
        raise NotImplementedError("Helion CuTe lowering found an unexpected launch ABI")
    # The steps Helion's own compile path runs before and after cute.compile.
    launcher._ensure_cute_dsl_arch_env(tensors)
    jit_function = launcher._create_cute_wrapper(
        captured.cute_kernel,
        schema,
        captured.block,
        num_sm=launcher._cute_num_sm_from_arch_args(tensors),
        device=tensors[0].device,
    )
    stream = launcher._cute_current_stream()
    options = " ".join(
        o for o in (captured.compile_options or "").split() if o != _TVM_FFI_OPTION
    )
    if options:
        compiled = cute.compile(jit_function, *launch_args, stream, options=options)
    else:
        compiled = cute.compile(jit_function, *launch_args, stream)
    # Helion sets _helion_cute_sass_checks only on kernels that declared SASS
    # checks, so a missing attribute means none were requested.
    sass_checks = tuple(getattr(captured.cute_kernel, "_helion_cute_sass_checks", ()))
    sass_check.verify_compiled(compiled, sass_checks, prefix)
    compiled.export_to_c(
        file_path=str(workdir), file_name=prefix, function_prefix=prefix
    )
    header = (workdir / f"{prefix}.h").read_text()
    _check_error_check_macro(header)
    expected = len(tensors) + 3
    signature = re.search(rf"cute_dsl_{prefix}_wrapper\(([^)]*)\)", header)
    if signature is None or len(signature.group(1).split(",")) != expected + 2:
        raise NotImplementedError("Helion CuTe lowering found an unexpected C wrapper")
    return compiled, alignments


def _build_library(
    prefix: str,
    workdir: Path,
    tensors: tuple[torch.Tensor, ...],
    alignments: list[int],
    grid: tuple[int, int, int],
) -> str:
    """Compile the C shim and archive it with the exported kernel object."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("Helion CuTe lowering needs a CUDA toolkit (CUDA_HOME)")
    shim = f"aoti_torch_cuda_{prefix}"
    count = len(tensors)
    parameters = ", ".join(f"AtenTensorHandle t{i}" for i in range(count))

    layout_arrays = "\n".join(
        f"constexpr int64_t k{name}{i}[] = {{{', '.join(map(str, values)) or '0'}}};"
        for i, t in enumerate(tensors)
        for name, values in (("Sizes", t.shape), ("Strides", t.stride()))
    )

    (workdir / "shim.cpp").write_text(
        _SHIM_TEMPLATE.format(
            prefix=prefix,
            shim=shim,
            parameters=parameters,
            count=count,
            ranks=", ".join(str(t.dim()) for t in tensors),
            layout_arrays=layout_arrays,
            sizes=", ".join(f"kSizes{i}" for i in range(count)),
            strides=", ".join(f"kStrides{i}" for i in range(count)),
            handles=", ".join(f"t{i}" for i in range(count)),
            alignments=", ".join(str(a) for a in alignments),
            pointer_args=", ".join(f"pointers[{i}]" for i in range(count)),
            grid=", ".join(str(g) for g in grid),
        )
    )
    try:
        subprocess.run(
            [
                get_cpp_compiler(),
                "-c",
                "-fPIC",
                "-O2",
                "-std=c++17",
                f"-I{workdir}",
                f"-I{CUDA_HOME}/include",
                str(workdir / "shim.cpp"),
                "-o",
                str(workdir / "shim.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                shutil.which("ar") or "ar",
                "rcs",
                str(workdir / f"lib{prefix}.a"),
                str(workdir / "shim.o"),
                str(workdir / f"{prefix}.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Helion CuTe lowering failed to build the launch library: "
            f"{error.stderr}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Helion CuTe lowering cannot run the C++ compiler or ar: {error}"
        ) from error
    return f"AOTITorchError {shim}({parameters}, AtenTensorHandle* ret)"


def _register_op(
    prefix: str,
    mutable: Sequence[bool],
    launch: Callable[[tuple[torch.Tensor, ...]], None],
) -> Any:
    arguments = ", ".join(
        f"Tensor(a{i}!) t{i}" if m else f"Tensor t{i}" for i, m in enumerate(mutable)
    )
    with _STATE.lock:
        if _STATE.library is None:
            _STATE.library = torch.library.Library(_OP_NAMESPACE, "FRAGMENT")
        library = _STATE.library
    # The kernel is compiled for the exported sizes and strides.
    library.define(
        f"{prefix}({arguments}) -> ()", tags=(torch.Tag.needs_exact_strides,)
    )

    def run(*tensors: torch.Tensor) -> None:
        launch(tensors)

    def fake(*tensors: torch.Tensor) -> None:
        return None

    library.impl(prefix, run, "CUDA")
    torch.library.register_fake(f"{_OP_NAMESPACE}::{prefix}", fake, lib=library)
    return getattr(getattr(torch.ops, _OP_NAMESPACE), prefix).default


def _build_launch_archive(
    prefix: str, captured: _CapturedLaunch
) -> tuple[Any, list[int], str, Path]:
    """Compile the launch and archive it, without registering any op."""
    # Build the shared dialect runtime now, so a build failure surfaces at
    # export instead of inside the AOTInductor link at the end of the compile.
    _runtime_archive_directory()
    root = Path(cache_dir()) / "executorch_helion_cute"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        compiled, alignments = _compile_and_export(captured, prefix, workdir)
        declaration = _build_library(
            prefix, workdir, captured.tensors, alignments, captured.grid
        )
        # A directory of its own per export, created only once the build
        # succeeded, so a process never links an archive another process
        # wrote under the same name.
        library_directory = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=root))
        shutil.copyfile(
            workdir / f"lib{prefix}.a", library_directory / f"lib{prefix}.a"
        )
    _STATE.track_library_directory(library_directory)
    return compiled, alignments, declaration, library_directory


def _export(
    prefix: str, captured: _CapturedLaunch, mutable: Sequence[bool]
) -> _ExportedLaunch:
    launcher = _cute_launcher_modules()[0]
    compiled, alignments, declaration, library_directory = _build_launch_archive(
        prefix, captured
    )
    cute_kernel, grid = captured.cute_kernel, captured.grid
    # Only layouts: keeping the example tensors themselves alive would pin
    # their GPU memory (weights included) for the life of the process.
    layouts = tuple((tuple(t.shape), t.stride(), t.dtype) for t in captured.tensors)

    def launch(tensors: tuple[torch.Tensor, ...]) -> None:
        # The eager twin of the C shim's check: same sizes, strides and dtype
        # as compiled for, and each pointer aligned at least as much as the
        # compiled alignment. An exactly matching Helion schema would also
        # reject a more-aligned pointer for an element-aligned operand, which
        # the shim (and the compiled kernel) accepts.
        if len(tensors) != len(layouts) or any(
            tuple(tensor.shape) != shape
            or any(
                actual != expected
                for actual, expected, size in zip(tensor.stride(), stride, shape)
                if size != 1
            )
            or tensor.dtype != dtype
            or tensor.data_ptr() % alignment != 0
            for tensor, (shape, stride, dtype), alignment in zip(
                tensors, layouts, alignments
            )
        ):
            raise RuntimeError(
                f"{prefix} was compiled for other argument layouts or alignments"
            )
        args = launcher._build_cute_schema_and_args(cute_kernel, tensors, grid)
        compiled(*args.launch_args, launcher._cute_current_stream())

    return _ExportedLaunch(
        _register_op(prefix, mutable, launch), declaration, prefix, library_directory
    )


def export_cute_launch(
    entrypoint: Callable[..., Any],
    launcher_parameter: str,
    source: str,
    fake_arguments: tuple[torch.Tensor, ...],
    mutated_arguments: frozenset[int],
    aligned_arguments: frozenset[int],
) -> tuple[Callable[..., None], Any]:
    """Export the single CuTe launch of ``entrypoint``.

    Returns a launcher with Helion's launcher signature that calls the custom
    op the launch was registered as, together with that op, so tracing the
    entrypoint with the launcher records the op in place of the launch and
    the caller can check for exactly that op.
    """
    with unset_fake_temporarily():
        arguments = tuple(
            _example_like(a, i in aligned_arguments)
            for i, a in enumerate(fake_arguments)
        )
        captured = _capture_launch(entrypoint, launcher_parameter, arguments)
        # The kernel writes the storages the host code allocated and those of
        # the inputs the captured call recorded as mutated.
        argument_of = {
            StorageWeakRef(a.untyped_storage()): i for i, a in enumerate(arguments)
        }
        mutable = []
        for tensor in captured.tensors:
            index = argument_of.get(StorageWeakRef(tensor.untyped_storage()))
            mutable.append(index is None or index in mutated_arguments)
        key = repr(
            (
                source,
                [
                    (
                        tuple(t.shape),
                        # Size-1 strides never affect an address (both
                        # runtime checks ignore them), so do not let them
                        # split one effective layout into two prefixes.
                        tuple(
                            1 if size == 1 else stride
                            for stride, size in zip(t.stride(), t.shape)
                        ),
                        t.storage_offset(),
                        t.dtype,
                    )
                    for t in captured.tensors
                ],
                captured.grid,
                captured.block,
                captured.compile_options,
                mutable,
                torch.cuda.get_device_capability(captured.tensors[0].device),
            )
        )
        prefix = f"helion_cute_{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        # The global lock guards only the exported map; compiling takes the
        # per-prefix lock, so other lowerings are not stuck behind this
        # kernel's CuTe compile and the same prefix is still built once.
        with _STATE.prefix_lock(prefix):
            with _STATE.lock:
                exported = _STATE.exported.get(prefix)
            if (
                exported is not None
                and not (
                    exported.library_directory / f"lib{exported.library}.a"
                ).is_file()
            ):
                # The cached archive is gone (for example its cache directory
                # was cleared under a long-lived process). Rebuild only the
                # archive: the op is still registered, and defining it again
                # would raise.
                _, _, _, directory = _build_launch_archive(prefix, captured)
                exported = exported._replace(library_directory=directory)
                with _STATE.lock:
                    _STATE.exported[prefix] = exported
            if exported is None:
                exported = _export(prefix, captured, mutable)
                with _STATE.lock:
                    _STATE.exported[prefix] = exported
    record_program_launch(exported)
    op = exported.op

    def launch_op(
        cute_kernel: Any, grid: Any, *args: torch.Tensor, **kwargs: Any
    ) -> None:
        op(*args)

    return launch_op, op


def begin_program() -> None:
    """Forget the launches of the previous program lowered on this thread."""
    _STATE.program_launches().clear()
    _STATE.per_thread.last_had_launches = False


def note_existing_launches(modules: Any) -> None:
    """Record launches a program already contains as this thread's launches.

    A program lowered by an earlier ``LowerHelionPass`` call holds
    ``helion_cute::`` ops but no Helion calls, so nothing new is exported
    when the backend lowers it; its archives must still link (and its PTX
    flag must still be suppressed).
    """
    with _STATE.lock:
        exported = list(_STATE.exported.values())
    if not exported:
        return
    for module in modules:
        for node in module.graph.nodes:
            # Edge conversion rewraps the op target, and functionalization
            # wraps a writing call in auto_functionalized with the op as its
            # first argument, so match the launch's unique prefix in either.
            target = str(node.target)
            if node.args:
                target += str(node.args[0])
            for launch in exported:
                if launch.library in target:
                    record_program_launch(launch)


def record_program_launch(exported: _ExportedLaunch) -> None:
    """Record a launch as part of this thread's program.

    Any number of distinct launches may be recorded: each was exported
    under its own prefix, registered as its own op and archived in its
    own library, and the link adds every recorded archive.
    """
    _STATE.program_launches()[exported.library] = exported


def last_program_had_launches() -> bool:
    """Whether this thread's last completed compile linked CuTe launches."""
    return _STATE.per_thread.last_had_launches


def current_program_launches() -> list[_ExportedLaunch]:
    """The launches recorded for this thread's program so far."""
    return list(_STATE.program_launches().values())


def exported_fallback_kernels() -> dict[str, None]:
    """This thread's program launches, as supported fallback kernels.

    The collector that reads this set records only into the compiling
    thread's collector, so a per-program set cannot fail another thread's
    compile, and an op the program contains but this compile never
    recorded (for example a scan miss) is reported missing at export time
    instead of linking a `.so` with an undefined shim symbol.
    """
    return {
        f"{_OP_NAMESPACE}::{launch.library}": None
        for launch in current_program_launches()
    }


def program_link_options() -> tuple[list[str], list[str]]:
    """Link directories and libraries for this thread's program launches.

    Only this compile's launches (exported or noted by the pass on this
    thread) are linked, so an unrelated compile never sees these archives.
    """
    launches = list(_STATE.program_launches().values())
    if not launches:
        return [], []
    missing = [
        str(e.library_directory / f"lib{e.library}.a")
        for e in launches
        if not (e.library_directory / f"lib{e.library}.a").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Helion CuTe launch archives are missing: " + ", ".join(missing)
        )
    directories = [str(e.library_directory) for e in launches]
    # The one shared dialect runtime archive, linked once per program.
    directories.append(str(_runtime_archive_directory()))
    libraries = [e.library for e in launches] + [_RUNTIME_LIBRARY]
    return directories, libraries


@contextlib.contextmanager
def link_exported_launches() -> Iterator[None]:
    """Scope this thread's program launches to one AOTInductor compile.

    Inductor's config is thread-local, so shim declarations written at
    export time are only visible on the exporting thread; (re)declare every
    exported shim on the compiling thread here. The archives are added by
    the CUDA backend's thread-scoped compile patch reading
    ``program_link_options``. On exit the thread's launches are forgotten,
    remembering only whether the compile had any
    (``last_program_had_launches``).
    """
    launches = list(_STATE.program_launches().values())
    with _STATE.lock:
        exported = list(_STATE.exported.values())
    shims = dict(torch._inductor.config.aot_inductor.custom_ops_to_c_shims)
    shims.update({e.op: [e.shim_declaration] for e in exported})
    try:
        if exported:
            with torch._inductor.config.patch(
                {"aot_inductor.custom_ops_to_c_shims": shims}
            ):
                yield
        else:
            yield
    finally:
        _STATE.per_thread.last_had_launches = bool(launches)
        _STATE.program_launches().clear()
