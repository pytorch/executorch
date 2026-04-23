/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI op-registry fallback impls.
//
// SlimTensor handles arrive from the AOTI .so. We materialize zero-copy
// CPU-side ETensor views for them (because MetalOpRegistry consumes
// EValue, which holds etensor::Tensor — not SlimTensor) and dispatch via
// the registry. The TensorPtr vectors keep the views alive across
// dispatch.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_ops.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalOp.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalOpRegistry.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>

#include <array>
#include <new>

namespace executorch {
namespace backends {
namespace metal {

namespace {

using executorch::backends::metal_v2::MetalOp;
using executorch::backends::metal_v2::MetalOpRegistry;

// Convert PyTorch dtype code → ExecuTorch's exec_aten ScalarType.
executorch::aten::ScalarType to_aten_scalar_type(
    executorch::backends::aoti::slim::c10::ScalarType slim_dt) {
  return static_cast<executorch::aten::ScalarType>(static_cast<int>(slim_dt));
}

// Maximum tensor rank we view in-place. Bump if any op exceeds this; today
// our ops are all ≤4D (mm/bmm/add/relu).
constexpr size_t kMaxTensorDim = 8;

// A single-tensor view materialized entirely on the stack: sizes, strides,
// dim_order arrays + a TensorImpl in placement-new storage. Construct via
// makeView(); resulting Tensor wraps the placement-new'd TensorImpl, so
// holding the StackTensorView in scope keeps the Tensor valid.
//
// This replaces extension::from_blob() which heap-allocates Storage +
// TensorImpl + 2 shared_ptr control blocks per call. Per-dispatch we now
// allocate zero heap.
struct StackTensorView {
  using SizesType = executorch::aten::SizesType;
  using StridesType = executorch::aten::StridesType;
  using DimOrderType = executorch::runtime::etensor::TensorImpl::DimOrderType;
  using TensorImpl = executorch::runtime::etensor::TensorImpl;
  using ETensor = executorch::runtime::etensor::Tensor;

  std::array<SizesType, kMaxTensorDim> sizes;
  std::array<StridesType, kMaxTensorDim> strides;
  std::array<DimOrderType, kMaxTensorDim> dim_order;
  alignas(TensorImpl) std::byte impl_storage[sizeof(TensorImpl)];
  bool constructed = false;

  // Construct the TensorImpl in-place over a SlimTensor's storage and
  // return a wrapping ETensor (cheap: just a TensorImpl* copy).
  ETensor makeView(const Tensor& s) {
    const size_t dim = static_cast<size_t>(s.dim());
    ET_CHECK_MSG(dim <= kMaxTensorDim,
        "StackTensorView: tensor rank %zu exceeds kMaxTensorDim=%zu",
        dim, kMaxTensorDim);
    for (size_t i = 0; i < dim; ++i) {
      sizes[i] = static_cast<SizesType>(s.sizes()[i]);
      strides[i] = static_cast<StridesType>(s.strides()[i]);
      dim_order[i] = static_cast<DimOrderType>(i);
    }
    auto* impl = new (impl_storage) TensorImpl(
        to_aten_scalar_type(s.dtype()),
        static_cast<ssize_t>(dim),
        sizes.data(),
        s.data_ptr(),
        dim_order.data(),
        strides.data(),
        // DYNAMIC_BOUND so resize_tensor (if invoked by an op) is a no-op
        // when the requested shape matches our current shape — matches the
        // prior from_blob default.
        executorch::runtime::TensorShapeDynamism::DYNAMIC_BOUND);
    constructed = true;
    return ETensor(impl);
  }

  ~StackTensorView() {
    if (constructed) {
      reinterpret_cast<TensorImpl*>(impl_storage)->~TensorImpl();
    }
  }

  StackTensorView() = default;
  StackTensorView(const StackTensorView&) = delete;
  StackTensorView& operator=(const StackTensorView&) = delete;
};

// Maximum number of in/out tensors any registry op consumes. We stack-allocate
// EValue + pointer storage sized to this so dispatch is heap-alloc-free.
// Bump if a future op needs more — small overhead per call.
constexpr size_t kMaxOpInputs = 8;
constexpr size_t kMaxOpOutputs = 4;

// Run a pre-resolved op against a set of SlimTensor handles. Materializes
// ETensor views (zero-copy, in-place via StackTensorView), wraps them as
// EValues, and dispatches. Zero heap allocations on the hot path: sizes,
// strides, dim_order, TensorImpl, EValue, EValue* pointer arrays — all
// stack-resident.
AOTITorchError dispatchOp(
    MetalOp* op,
    std::initializer_list<Tensor*> inTensors,
    std::initializer_list<Tensor*> outTensors) {
  if (!op) return Error::NotImplemented;
  if (inTensors.size() > kMaxOpInputs ||
      outTensors.size() > kMaxOpOutputs) {
    ET_LOG(Error,
        "aoti_ops_v2: op '%s' exceeds max in=%zu/%zu, out=%zu/%zu",
        op->name(), inTensors.size(), kMaxOpInputs,
        outTensors.size(), kMaxOpOutputs);
    return Error::InvalidArgument;
  }

  // Stack-resident ETensor views (each owns sizes/strides/dim_order arrays
  // and a TensorImpl in placement-new storage). Holding these in scope
  // keeps the ETensors valid for the duration of dispatch.
  std::array<StackTensorView, kMaxOpInputs> inViews;
  std::array<StackTensorView, kMaxOpOutputs> outViews;
  std::array<executorch::runtime::EValue, kMaxOpInputs> inEValues;
  std::array<executorch::runtime::EValue, kMaxOpOutputs> outEValues;
  std::array<executorch::runtime::EValue*, kMaxOpInputs> inPtrs;
  std::array<executorch::runtime::EValue*, kMaxOpOutputs> outPtrs;

  size_t in_n = 0;
  for (auto* t : inTensors) {
    inEValues[in_n] = executorch::runtime::EValue(inViews[in_n].makeView(*t));
    inPtrs[in_n] = &inEValues[in_n];
    ++in_n;
  }
  size_t out_n = 0;
  for (auto* t : outTensors) {
    outEValues[out_n] =
        executorch::runtime::EValue(outViews[out_n].makeView(*t));
    outPtrs[out_n] = &outEValues[out_n];
    ++out_n;
  }

  op->dispatch(
      getMetalStream(),
      MetalOp::EValuePtrSpan(inPtrs.data(), in_n),
      MetalOp::EValuePtrSpan(outPtrs.data(), out_n));
  return Error::Ok;
}

// Convenience wrapper: does the (string-keyed, allocating) registry lookup
// each call. Prefer dispatchOp(MetalOp*, ...) at hot call sites where the
// op name is fixed — see the static-cached pattern in aoti_torch_mps_mm_out.
AOTITorchError dispatchRegistryOp(
    const char* opName,
    std::initializer_list<Tensor*> inTensors,
    std::initializer_list<Tensor*> outTensors) {
  auto* op = MetalOpRegistry::shared().get(opName);
  if (!op) {
    ET_LOG(Error, "aoti_ops_v2: op '%s' not found in MetalOpRegistry", opName);
    return Error::NotImplemented;
  }
  return dispatchOp(op, inTensors, outTensors);
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  if (!out || !self || !mat2) return Error::InvalidArgument;
  // Cache the op pointer across calls to skip the registry lookup hot path
  // (which hashes a std::string("aten::mm") on every call).
  static MetalOp* op = MetalOpRegistry::shared().get("aten::mm");
  return dispatchOp(
      op,
      {reinterpret_cast<Tensor*>(self), reinterpret_cast<Tensor*>(mat2)},
      {reinterpret_cast<Tensor*>(out)});
}

AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  if (!out || !self || !mat2) return Error::InvalidArgument;
  static MetalOp* op = MetalOpRegistry::shared().get("aten::bmm");
  return dispatchOp(
      op,
      {reinterpret_cast<Tensor*>(self), reinterpret_cast<Tensor*>(mat2)},
      {reinterpret_cast<Tensor*>(out)});
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
