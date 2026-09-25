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
#include <string>

namespace executorch::backends::webgpu {

constexpr uint32_t kTensorMetaMaxNdim = 8;

// Per-tensor metadata UBO; mirrors Vulkan BufferMetadata (8-dim NCHW, std140).
struct TensorMeta {
  uint32_t ndim;
  uint32_t numel;
  uint32_t _pad[2];
  uint32_t sizes[kTensorMetaMaxNdim];
  uint32_t strides[kTensorMetaMaxNdim];
};

static_assert(
    sizeof(TensorMeta) == 80,
    "TensorMeta std140 layout must be 80 bytes to match the WGSL uniform");
// Lock the std140 field offsets the WGSL uniform reads, not just total size.
static_assert(offsetof(TensorMeta, ndim) == 0);
static_assert(offsetof(TensorMeta, numel) == 4);
static_assert(offsetof(TensorMeta, sizes) == 16);
static_assert(offsetof(TensorMeta, strides) == 48);

// Fill TensorMeta from NCHW dims: contiguous strides, padded trailing slots.
inline void fill_tensor_meta(const WebGPUTensor& t, TensorMeta* m) {
  const uint32_t ndim = static_cast<uint32_t>(t.dims.size());
  if (ndim > kTensorMetaMaxNdim) {
    throw std::runtime_error("TensorMeta: tensor rank exceeds 8 (MAX_NDIM)");
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
    throw std::runtime_error("TensorMeta: out_ndim exceeds 8 (MAX_NDIM)");
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

// Validate the operands of an elementwise binary op and report whether the i32
// shader variant is required.
//
// Integer operands must never reach an f32 shader. Both dtypes are 4 bytes, so
// a size-only check lets int data through, and the f32 shader then reinterprets
// the bit pattern: a small integer becomes a denormal, so the result is
// silently ~0 rather than visibly wrong. Mixed int/float operands and integer
// widths the shaders cannot address (bool, int8, and non-downcast int64) are
// rejected outright.
inline bool binary_operands_are_int(
    const WebGPUTensor& in1,
    const WebGPUTensor& in2,
    const WebGPUTensor& out,
    const TensorMeta& in1_meta,
    const TensorMeta& in2_meta,
    const TensorMeta& out_meta,
    const char* op_name) {
  const std::string name(op_name);
  const bool is_int = out.is_int;
  if (in1.is_int != is_int || in2.is_int != is_int) {
    throw std::runtime_error(name + ": mixed integer and float operands");
  }
  if (is_int &&
      (in1.is_bool || in2.is_bool || out.is_bool || in1.is_int8 ||
       in2.is_int8 || out.is_int8)) {
    throw std::runtime_error(name + ": bool and int8 operands are unsupported");
  }
  // 4 bytes per element for both f32 and i32; an 8-byte int64 that was not
  // downcast by the AoT stack lands here and is rejected.
  if (out.nbytes != static_cast<size_t>(out_meta.numel) * 4u ||
      in1.nbytes != static_cast<size_t>(in1_meta.numel) * 4u ||
      in2.nbytes != static_cast<size_t>(in2_meta.numel) * 4u) {
    throw std::runtime_error(
        name + ": operand is not 4 bytes per element (nbytes != numel * 4)");
  }
  return is_int;
}

} // namespace executorch::backends::webgpu
