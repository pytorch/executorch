/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "BinaryOps.h"
#include <executorch/backends/portable/runtime/metal_v2/kernels/Accessors.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/runtime/platform/log.h>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::get_broadcast_target_size;

//===----------------------------------------------------------------------===//
// Variant detection, prefix string, and broadcast strides come from
// metal_v2/OpUtils.h. Kernel-name builder is the only thing left here.
//===----------------------------------------------------------------------===//

std::string BinaryOp::kernelName(ElementwiseVariant variant, ScalarType dtype) const {
  return std::string(variantPrefix(variant)) + "_" + opName() + "_" + dtypeSuffix(dtype);
}

//===----------------------------------------------------------------------===//
// Output Shape Computation - using ET's broadcast utility
//===----------------------------------------------------------------------===//

std::vector<SizesType> BinaryOp::computeOutputShape(
    EValuePtrSpan inputs) const {

  if (inputs.size() < 2 || !inputs[0]->isTensor() || !inputs[1]->isTensor()) {
    return {};
  }

  auto& a = inputs[0]->toTensor();
  auto& b = inputs[1]->toTensor();

  // Use ET's broadcast utility
  SizesType out_sizes[runtime::kTensorDimensionLimit];
  size_t out_dim = 0;

  Error err = get_broadcast_target_size(a, b, out_sizes, runtime::kTensorDimensionLimit, &out_dim);
  if (err != Error::Ok) {
    return {};
  }

  return std::vector<SizesType>(out_sizes, out_sizes + out_dim);
}

// (broadcastStrides + collapseContiguousDims live in OpUtils.h.)

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

void BinaryOp::dispatch(
    MetalStream* stream,
    EValuePtrSpan inputs,
    EValuePtrSpan outputs) {

  auto& a = inputs[0]->toTensor();
  auto& b = inputs[1]->toTensor();
  auto& out = outputs[0]->toTensor();

  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "BinaryOp: failed to resize output");
    return;
  }

  ScalarType dtype = out.scalar_type();
  ElementwiseVariant variant = classifyBinary(a, b);
  std::string kname = kernelName(variant, dtype);

  auto* kernel = getKernel(stream, kname.c_str());
  size_t numel = out.numel();

  ET_LOG(Info, "BinaryOp::dispatch(%s): variant=%s, kernel=%s, numel=%zu",
         name(), variantPrefix(variant), kname.c_str(), numel);

  constexpr uint32_t blockSize = 256;
  // Kernels are templated on N = WorkPerThread<T>::n; pick the matching N
  // here so the host launch matches the kernel's per-thread work.
  const uint32_t elemPerThread = static_cast<uint32_t>(workPerThread(dtype));

  switch (variant) {
    case ElementwiseVariant::ScalarScalar: {
      if (hasAlpha()) {
        float alpha = 1.0f;
        stream->dispatch(kernel, {
          {a.mutable_data_ptr(), a.nbytes()},
          {b.mutable_data_ptr(), b.nbytes()},
          {out.mutable_data_ptr(), out.nbytes()},
          alpha,
          static_cast<uint32_t>(numel)
        }, uvec3(1, 1, 1), uvec3(1, 1, 1));
      } else {
        stream->dispatch(kernel, {
          {a.mutable_data_ptr(), a.nbytes()},
          {b.mutable_data_ptr(), b.nbytes()},
          {out.mutable_data_ptr(), out.nbytes()},
          static_cast<uint32_t>(numel)
        }, uvec3(1, 1, 1), uvec3(1, 1, 1));
      }
      break;
    }

    case ElementwiseVariant::ScalarVector:
    case ElementwiseVariant::VectorScalar:
    case ElementwiseVariant::VectorVector: {
      uint32_t gridX = (uint32_t)((numel + elemPerThread * blockSize - 1) / (elemPerThread * blockSize));

      if (hasAlpha()) {
        float alpha = 1.0f;
        stream->dispatch(kernel, {
          {a.mutable_data_ptr(), a.nbytes()},
          {b.mutable_data_ptr(), b.nbytes()},
          {out.mutable_data_ptr(), out.nbytes()},
          alpha,
          static_cast<uint32_t>(numel)
        }, uvec3(gridX, 1, 1), uvec3(blockSize, 1, 1));
      } else {
        stream->dispatch(kernel, {
          {a.mutable_data_ptr(), a.nbytes()},
          {b.mutable_data_ptr(), b.nbytes()},
          {out.mutable_data_ptr(), out.nbytes()},
          static_cast<uint32_t>(numel)
        }, uvec3(gridX, 1, 1), uvec3(blockSize, 1, 1));
      }
      break;
    }

    case ElementwiseVariant::General: {
      auto out_shape = computeOutputShape(inputs);
      auto a_strides_full = broadcastStrides(a.sizes(), out_shape);
      auto b_strides_full = broadcastStrides(b.sizes(), out_shape);

      // Collapse adjacent dims that are contiguous in BOTH inputs to
      // shrink ndim and reduce per-element index math in the shader.
      auto [shape_collapsed, strides_collapsed] =
          collapseContiguousDims(out_shape, {a_strides_full, b_strides_full});
      int32_t ndim = static_cast<int32_t>(shape_collapsed.size());

      std::vector<int32_t> shape_i32(shape_collapsed.begin(), shape_collapsed.end());
      std::vector<int32_t> a_strides_i32(strides_collapsed[0].begin(), strides_collapsed[0].end());
      std::vector<int32_t> b_strides_i32(strides_collapsed[1].begin(), strides_collapsed[1].end());

      uint32_t gridX = (uint32_t)((numel + blockSize - 1) / blockSize);

      stream->dispatch(kernel, {
        {a.mutable_data_ptr(), a.nbytes()},
        {b.mutable_data_ptr(), b.nbytes()},
        {out.mutable_data_ptr(), out.nbytes()},
        {shape_i32.data(), shape_i32.size() * sizeof(int32_t)},
        {a_strides_i32.data(), a_strides_i32.size() * sizeof(int32_t)},
        {b_strides_i32.data(), b_strides_i32.size() * sizeof(int32_t)},
        ndim,
        static_cast<uint32_t>(numel)
      }, uvec3(gridX, 1, 1), uvec3(blockSize, 1, 1));
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Kernel Source
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Kernel Source
//
// We prepend metal_v2/kernels/Accessors.h's shared shader helpers (elemToLoc
// family) to the per-op kernel body. The result is built once into a static
// std::string and returned by .c_str() so that MetalOp's `const char*` API is
// preserved.
//===----------------------------------------------------------------------===//

const char* BinaryOp::kernelSource() const {
  static const std::string source = std::string(kAccessorsMetalSource) + R"(
#include <metal_stdlib>
using namespace metal;

// Op Functors
struct AddOp {
  template<typename T> T operator()(T a, T b) { return a + b; }
  template<typename T> T operator()(T a, T b, float alpha) { return a + T(alpha) * b; }
};
struct MulOp { template<typename T> T operator()(T a, T b) { return a * b; } };
struct SubOp { template<typename T> T operator()(T a, T b) { return a - b; } };

// ScalarScalar (ss)
template<typename T, typename Op>
kernel void binary_ss(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid == 0) out[0] = Op()(a[0], b[0]);
}

template<typename T, typename Op>
kernel void binary_ss_alpha(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid == 0) out[0] = Op()(a[0], b[0], alpha);
}

// ScalarVector (sv) -- N elements per thread, dtype-aware.
template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_sv(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  T scalar = a[0];
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(scalar, b[j]);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(scalar, b[idx + i]);
  }
}

template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_sv_alpha(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  T scalar = a[0];
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(scalar, b[j], alpha);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(scalar, b[idx + i], alpha);
  }
}

// VectorScalar (vs) -- N elements per thread, dtype-aware.
template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_vs(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  T scalar = b[0];
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(a[j], scalar);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(a[idx + i], scalar);
  }
}

template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_vs_alpha(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  T scalar = b[0];
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(a[j], scalar, alpha);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(a[idx + i], scalar, alpha);
  }
}

// VectorVector (vv) -- N elements per thread, dtype-aware.
template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_vv(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(a[j], b[j]);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(a[idx + i], b[idx + i]);
  }
}

template<typename T, typename Op, int N = WorkPerThread<T>::n>
kernel void binary_vv_alpha(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint idx = gid * uint(N);
  if (N > 1 && idx + uint(N) > n) {
    for (uint j = idx; j < n; j++) out[j] = Op()(a[j], b[j], alpha);
  } else {
    for (int i = 0; i < N; ++i) out[idx + i] = Op()(a[idx + i], b[idx + i], alpha);
  }
}

// General (g) - Strided/broadcast.
// Uses elemToLocBinary from Accessors.h: decodes both inputs in one loop.
template<typename T, typename Op>
kernel void binary_g(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant int* a_strides [[buffer(4)]],
    constant int* b_strides [[buffer(5)]],
    constant int& ndim [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  int2 idx = elemToLocBinary(gid, shape, a_strides, b_strides, ndim);
  out[gid] = Op()(a[idx.x], b[idx.y]);
}

// Instantiate
#define INSTANTIATE_SS(name, op, T, suffix) \
  template [[host_name("ss_" name "_" suffix)]] kernel void binary_ss<T, op>(device const T*, device const T*, device T*, constant uint&, uint);

#define INSTANTIATE_SS_ALPHA(name, op, T, suffix) \
  template [[host_name("ss_" name "_" suffix)]] kernel void binary_ss_alpha<T, op>(device const T*, device const T*, device T*, constant float&, constant uint&, uint);

#define INSTANTIATE_SV(name, op, T, suffix) \
  template [[host_name("sv_" name "_" suffix)]] kernel void binary_sv<T, op>(device const T*, device const T*, device T*, constant uint&, uint);

#define INSTANTIATE_SV_ALPHA(name, op, T, suffix) \
  template [[host_name("sv_" name "_" suffix)]] kernel void binary_sv_alpha<T, op>(device const T*, device const T*, device T*, constant float&, constant uint&, uint);

#define INSTANTIATE_VS(name, op, T, suffix) \
  template [[host_name("vs_" name "_" suffix)]] kernel void binary_vs<T, op>(device const T*, device const T*, device T*, constant uint&, uint);

#define INSTANTIATE_VS_ALPHA(name, op, T, suffix) \
  template [[host_name("vs_" name "_" suffix)]] kernel void binary_vs_alpha<T, op>(device const T*, device const T*, device T*, constant float&, constant uint&, uint);

#define INSTANTIATE_VV(name, op, T, suffix) \
  template [[host_name("vv_" name "_" suffix)]] kernel void binary_vv<T, op>(device const T*, device const T*, device T*, constant uint&, uint);

#define INSTANTIATE_VV_ALPHA(name, op, T, suffix) \
  template [[host_name("vv_" name "_" suffix)]] kernel void binary_vv_alpha<T, op>(device const T*, device const T*, device T*, constant float&, constant uint&, uint);

#define INSTANTIATE_G(name, op, T, suffix) \
  template [[host_name("g_" name "_" suffix)]] kernel void binary_g<T, op>(device const T*, device const T*, device T*, constant int*, constant int*, constant int*, constant int&, constant uint&, uint);

// Add (with alpha)
INSTANTIATE_SS_ALPHA("add", AddOp, float, "f32")
INSTANTIATE_SS_ALPHA("add", AddOp, half, "f16")
INSTANTIATE_SV_ALPHA("add", AddOp, float, "f32")
INSTANTIATE_SV_ALPHA("add", AddOp, half, "f16")
INSTANTIATE_VS_ALPHA("add", AddOp, float, "f32")
INSTANTIATE_VS_ALPHA("add", AddOp, half, "f16")
INSTANTIATE_VV_ALPHA("add", AddOp, float, "f32")
INSTANTIATE_VV_ALPHA("add", AddOp, half, "f16")
INSTANTIATE_G("add", AddOp, float, "f32")
INSTANTIATE_G("add", AddOp, half, "f16")

// Mul
INSTANTIATE_SS("mul", MulOp, float, "f32")
INSTANTIATE_SS("mul", MulOp, half, "f16")
INSTANTIATE_SV("mul", MulOp, float, "f32")
INSTANTIATE_SV("mul", MulOp, half, "f16")
INSTANTIATE_VS("mul", MulOp, float, "f32")
INSTANTIATE_VS("mul", MulOp, half, "f16")
INSTANTIATE_VV("mul", MulOp, float, "f32")
INSTANTIATE_VV("mul", MulOp, half, "f16")
INSTANTIATE_G("mul", MulOp, float, "f32")
INSTANTIATE_G("mul", MulOp, half, "f16")

// Sub
INSTANTIATE_SS("sub", SubOp, float, "f32")
INSTANTIATE_SS("sub", SubOp, half, "f16")
INSTANTIATE_SV("sub", SubOp, float, "f32")
INSTANTIATE_SV("sub", SubOp, half, "f16")
INSTANTIATE_VS("sub", SubOp, float, "f32")
INSTANTIATE_VS("sub", SubOp, half, "f16")
INSTANTIATE_VV("sub", SubOp, float, "f32")
INSTANTIATE_VV("sub", SubOp, half, "f16")
INSTANTIATE_G("sub", SubOp, float, "f32")
INSTANTIATE_G("sub", SubOp, half, "f16")
)";
  return source.c_str();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
