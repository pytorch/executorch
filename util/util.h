#pragma once

#include <algorithm>

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/runtime/executor/executor.h>
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

// Initialize input tensor from execution plan, returns an array of data
// pointers that were allocated and need to be freed
inline exec_aten::ArrayRef<void*> PrepareInputTensors(
    const ExecutionPlan& plan) {
  size_t input_size = plan.inputs_size();
  size_t num_allocated = 0;
  void** inputs = (void**)malloc(input_size * sizeof(void*));
#ifdef USE_ATEN_LIB
  auto deleteByNone = [](void* p) {};
  for (size_t i = 0; i < input_size; i++) {
    if (!plan.get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = plan.get_input(i).toTensor();
    at::StorageImpl* storage =
        t.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();
    auto& data_ptr = storage->data_ptr();
    if (data_ptr.get() == nullptr) {
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
    if (!plan.get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = plan.get_input(i).toTensor();
    if (t.data_ptr() == nullptr) {
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

// Free input pointers
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
