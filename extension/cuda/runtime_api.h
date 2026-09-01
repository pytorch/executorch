/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#if defined(EXECUTORCH_USE_HIP)

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemoryType = hipMemoryType;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaStream_t = hipStream_t;

inline constexpr cudaError_t cudaSuccess = hipSuccess;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice =
    hipMemcpyDeviceToDevice;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
inline constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
inline constexpr cudaMemoryType cudaMemoryTypeDevice = hipMemoryTypeDevice;
inline constexpr cudaMemoryType cudaMemoryTypeManaged = hipMemoryTypeManaged;
inline constexpr cudaStreamCaptureMode cudaStreamCaptureModeRelaxed =
    hipStreamCaptureModeRelaxed;
inline constexpr unsigned long long cudaGraphInstantiateFlagAutoFreeOnLaunch =
    hipGraphInstantiateFlagAutoFreeOnLaunch;

// A macro, unlike the aliases above, because hipStreamPerThread casts an
// integer to a pointer type and so is not a constant expression.
#define cudaStreamPerThread hipStreamPerThread

struct cudaPointerAttributes {
  cudaMemoryType type{};
  int device = -1;
};

inline cudaError_t cudaDeviceSynchronize() {
  return hipDeviceSynchronize();
}

inline cudaError_t cudaFree(void* ptr) {
  return hipFree(ptr);
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
  return hipFreeAsync(ptr, stream);
}

inline cudaError_t cudaGetDevice(int* device) {
  return hipGetDevice(device);
}

inline cudaError_t cudaGetDeviceCount(int* count) {
  return hipGetDeviceCount(count);
}

inline const char* cudaGetErrorString(cudaError_t error) {
  return hipGetErrorString(error);
}

inline cudaError_t cudaGetLastError() {
  return hipGetLastError();
}

inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  return hipGraphDestroy(graph);
}

inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec) {
  return hipGraphExecDestroy(graph_exec);
}

inline cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* graph_exec,
    cudaGraph_t graph,
    unsigned long long flags) {
  return hipGraphInstantiateWithFlags(graph_exec, graph, flags);
}

inline cudaError_t cudaGraphLaunch(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream) {
  return hipGraphLaunch(graph_exec, stream);
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
  return hipMalloc(ptr, size);
}

inline cudaError_t
cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return hipMallocAsync(ptr, size, stream);
}

inline cudaError_t
cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
  return hipMemcpy(dst, src, size, kind);
}

inline cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t size,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  return hipMemcpyAsync(dst, src, size, kind, stream);
}

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return hipMemGetInfo(free, total);
}

inline cudaError_t cudaPointerGetAttributes(
    cudaPointerAttributes* attributes,
    const void* ptr) {
  hipPointerAttribute_t hip_attributes{};
  const auto error = hipPointerGetAttributes(&hip_attributes, ptr);
  if (error == hipSuccess) {
    attributes->type = hip_attributes.type;
    attributes->device = hip_attributes.device;
  }
  return error;
}

inline cudaError_t cudaSetDevice(int device) {
  return hipSetDevice(device);
}

inline cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode) {
  return hipStreamBeginCapture(stream, mode);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  return hipStreamCreate(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return hipStreamDestroy(stream);
}

inline cudaError_t cudaStreamEndCapture(
    cudaStream_t stream,
    cudaGraph_t* graph) {
  return hipStreamEndCapture(stream, graph);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return hipStreamSynchronize(stream);
}

#else

#include <cuda_runtime.h>

#endif
