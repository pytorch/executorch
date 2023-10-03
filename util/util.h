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
inline exec_aten::ArrayRef<void*> PrepareInputTensors(const Method& method) {
  size_t input_size = method.inputs_size();
  size_t num_allocated = 0;
  void** inputs = (void**)malloc(input_size * sizeof(void*));
#ifdef USE_ATEN_LIB
  auto deleteByNone = [](void* p) {};
  for (size_t i = 0; i < input_size; i++) {
    if (!method.get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = method.get_input(i).toTensor();
    at::StorageImpl* storage =
        t.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();
    if (storage->data_ptr().get() == nullptr) {
      ET_LOG(Info, "input not initialized.");
      inputs[num_allocated++] = malloc(t.nbytes());
      storage->set_data_ptr(at::DataPtr(
          inputs[num_allocated - 1],
          inputs[num_allocated - 1],
          deleteByNone,
          DeviceType::CPU));
      storage->set_nbytes(t.nbytes());
    } else {
      ET_LOG(Info, "input already initialized, refilling.");
    }
    t.fill_(1.0f);
  }
#else
  for (size_t i = 0; i < input_size; i++) {
    if (!method.get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = method.get_input(i).toTensor();
    if (t.const_data_ptr() == nullptr) {
      ET_LOG(Info, "input not initialized.");
      inputs[num_allocated++] = malloc(t.nbytes());
      t.set_data(inputs[num_allocated - 1]);
    } else {
      ET_LOG(Info, "input already initialized, refilling.");
    }
    FillOnes(t);
  }
#endif
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
