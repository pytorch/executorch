/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

constexpr uint32_t kTensorMetaMaxNdim = 4;

// Per-tensor metadata UBO; mirrors Vulkan BufferMetadata (4-dim NCHW, std140).
struct TensorMeta {
  uint32_t ndim;
  uint32_t numel;
  uint32_t _pad[2];
  uint32_t sizes[kTensorMetaMaxNdim];
  uint32_t strides[kTensorMetaMaxNdim];
};

static_assert(
    sizeof(TensorMeta) == 48,
    "TensorMeta std140 layout must be 48 bytes to match the WGSL uniform");
// Lock the std140 field offsets the WGSL uniform reads, not just total size.
static_assert(offsetof(TensorMeta, ndim) == 0);
static_assert(offsetof(TensorMeta, numel) == 4);
static_assert(offsetof(TensorMeta, sizes) == 16);
static_assert(offsetof(TensorMeta, strides) == 32);

// Fill TensorMeta from NCHW dims: contiguous strides, padded trailing slots.
inline void fill_tensor_meta(const WebGPUTensor& t, TensorMeta* m) {
  const uint32_t ndim = static_cast<uint32_t>(t.dims.size());
  if (ndim > kTensorMetaMaxNdim) {
    throw std::runtime_error("TensorMeta: tensor rank exceeds 4 (MAX_NDIM)");
  }
  *m = {};
  for (uint32_t d = 0; d < kTensorMetaMaxNdim; d++) {
    m->sizes[d] = 1u;
    m->strides[d] = 0u;
  }
  m->ndim = ndim;
  uint32_t numel = 1u;
  uint32_t acc = 1u;
  for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
    const uint32_t sz = static_cast<uint32_t>(t.dims[i]);
    m->sizes[i] = sz;
    m->strides[i] = acc;
    acc *= sz;
    numel *= sz;
  }
  m->numel = numel;
}

// Broadcast variant: right-align operand dims into out rank (PyTorch trailing).
inline void fill_tensor_meta_broadcast(
    const WebGPUTensor& t,
    uint32_t out_ndim,
    TensorMeta* m) {
  const uint32_t rank = static_cast<uint32_t>(t.dims.size());
  if (out_ndim > kTensorMetaMaxNdim) {
    throw std::runtime_error("TensorMeta: out_ndim exceeds 4 (MAX_NDIM)");
  }
  if (rank > out_ndim) {
    throw std::runtime_error("TensorMeta: operand rank exceeds out_ndim");
  }
  *m = {};
  for (uint32_t d = 0; d < kTensorMetaMaxNdim; d++) {
    m->sizes[d] = 1u;
    m->strides[d] = 0u;
  }
  m->ndim = out_ndim;
  uint32_t acc = 1u;
  uint32_t numel = 1u;
  for (int i = static_cast<int>(rank) - 1; i >= 0; i--) {
    const uint32_t slot = out_ndim - rank + static_cast<uint32_t>(i);
    const uint32_t sz = static_cast<uint32_t>(t.dims[i]);
    m->sizes[slot] = sz;
    m->strides[slot] = acc;
    acc *= sz;
    numel *= sz;
  }
  m->numel = numel;
}

} // namespace executorch::backends::webgpu
