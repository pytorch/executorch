/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// StackTensorView — heap-free per-dispatch tensor view.
//
// Holds a placement-new'd TensorImpl over stack storage. Lets registry-op
// dispatch build an EValue(Tensor) over a SlimTensor without heap-allocating
// Storage + TensorImpl + shared_ptr control blocks per call.
//
// Usage:
//   StackTensorView v;
//   EValue ev(v.makeView(*tensor_handle));   // v must outlive ev

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_dtype.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/assert.h>

#include <array>
#include <cstddef>
#include <new>

namespace executorch {
namespace backends {
namespace metal {

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
    // DYNAMIC_BOUND: lets ops invoke resize_tensor as a no-op when the
    // requested shape matches the current shape.
    auto* impl = new (impl_storage) TensorImpl(
        to_aten_scalar_type(s.dtype()),
        static_cast<ssize_t>(dim),
        sizes.data(),
        s.data_ptr(),
        dim_order.data(),
        strides.data(),
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

}  // namespace metal
}  // namespace backends
}  // namespace executorch
