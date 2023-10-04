/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>
#ifdef USE_ATEN_LIB
#include <ATen/ATen.h> // @manual=//caffe2/aten:ATen-core
#endif

namespace torch {
namespace executor {
namespace util {

using namespace exec_aten;

// This macro defines all the scalar types that we currently support to fill a
// tensor. FillOnes() is a quick and dirty util that allows us to quickly
// initialize tensors and run a model.
#define EX_SCALAR_TYPES_SUPPORTED_BY_FILL(_fill_case) \
  _fill_case(uint8_t, Byte) /* 0 */                   \
      _fill_case(int8_t, Char) /* 1 */                \
      _fill_case(int16_t, Short) /* 2 */              \
      _fill_case(int, Int) /* 3 */                    \
      _fill_case(int64_t, Long) /* 4 */               \
      _fill_case(float, Float) /* 6 */                \
      _fill_case(double, Double) /* 7 */              \
      _fill_case(bool, Bool) /* 11 */

#define FILL_CASE(T, n)                                \
  case (ScalarType::n):                                \
    std::fill(                                         \
        tensor.mutable_data_ptr<T>(),                  \
        tensor.mutable_data_ptr<T>() + tensor.numel(), \
        1);                                            \
    break;

#ifndef USE_ATEN_LIB
inline void FillOnes(Tensor tensor) {
  switch (tensor.scalar_type()) {
    EX_SCALAR_TYPES_SUPPORTED_BY_FILL(FILL_CASE)
    default:
      ET_CHECK_MSG(false, "Scalar type is not supported by fill.");
  }
}
#endif

/**
 * Allocates input tensors for the provided Method, filling them with ones.
 *
 * @param[in] method The Method that owns the inputs to prepare.
 * @returns An array of pointers that must be passed to `FreeInputs()` after
 *     the Method is no longer needed.
 */
inline exec_aten::ArrayRef<void*> PrepareInputTensors(Method& method) {
  auto method_meta = method.method_meta();
  size_t input_size = method.inputs_size();
  size_t num_allocated = 0;
  void** inputs = (void**)malloc(input_size * sizeof(void*));

  for (size_t i = 0; i < input_size; i++) {
    if (*method_meta.input_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }

    // Tensor Input. Grab meta data and allocate buffer
    auto tensor_meta = method_meta.input_tensor_meta(i);
    inputs[num_allocated++] = malloc(tensor_meta->nbytes());

#ifdef USE_ATEN_LIB
    std::vector<int64_t> at_tensor_sizes;
    for (auto s : tensor_meta->sizes()) {
      at_tensor_sizes.push_back(s);
    }
    at::Tensor t = at::from_blob(
        inputs[num_allocated - 1],
        at_tensor_sizes,
        at::TensorOptions(tensor_meta->scalar_type()));
    t.fill_(1.0f);

#else // Portable Tensor
    // The only memory that needs to persist after set_input is called is the
    // data ptr of the input tensor, and that is only if the Method did not
    // memory plan buffer space for the inputs and instead is expecting the user
    // to provide them. Meta data like sizes and dim order are used to ensure
    // the input aligns with the values expected by the plan, but references to
    // them are not held onto.

    TensorImpl::SizesType* sizes = static_cast<TensorImpl::SizesType*>(
        malloc(sizeof(TensorImpl::SizesType) * tensor_meta->sizes().size()));
    TensorImpl::DimOrderType* dim_order =
        static_cast<TensorImpl::DimOrderType*>(malloc(
            sizeof(TensorImpl::DimOrderType) *
            tensor_meta->dim_order().size()));

    for (size_t size_idx = 0; size_idx < tensor_meta->sizes().size();
         size_idx++) {
      sizes[size_idx] = tensor_meta->sizes()[size_idx];
    }
    for (size_t dim_idx = 0; dim_idx < tensor_meta->dim_order().size();
         dim_idx++) {
      dim_order[dim_idx] = tensor_meta->dim_order()[dim_idx];
    }

    TensorImpl impl = TensorImpl(
        tensor_meta->scalar_type(),
        tensor_meta->sizes().size(),
        sizes,
        inputs[num_allocated - 1],
        dim_order);
    Tensor t(&impl);
    FillOnes(t);
#endif
    auto error = method.set_input(t, i);
    ET_CHECK_MSG(
        error == Error::Ok,
        "Error: 0x%" PRIx32 " setting input %zu.",
        error,
        i);
#ifndef USE_ATEN_LIB // Portable Tensor
    free(sizes);
    free(dim_order);
#endif
  }
  return {inputs, num_allocated};
}

/**
 * Frees memory that was allocated by `PrepareInputTensors()`.
 */
inline void FreeInputs(exec_aten::ArrayRef<void*> inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    free(inputs[i]);
  }
  free((void*)inputs.data());
}

#undef FILL_VALUE
#undef EX_SCALAR_TYPES_SUPPORTED_BY_FILL

} // namespace util
} // namespace executor
} // namespace torch
