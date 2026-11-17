/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Kernels that test_pybindings.py passes to load_method the way the
// ExecuTorchProgram.load_method docs show: from a library that exports them,
// loaded with ctypes. Its aten::add.out subtracts, so that a test can tell
// that it ran instead of the registered kernel.

#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include <cstddef>
#include <iterator>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)
void subtract_out(KernelRuntimeContext& context, Span<EValue*> stack) {
  const Tensor& self = stack[0]->toTensor();
  const Tensor& other = stack[1]->toTensor();
  Tensor& out = stack[3]->toTensor();
  const auto fits = [&out](const Tensor& tensor) {
    return tensor.scalar_type() == ScalarType::Float &&
        tensor.numel() == out.numel();
  };
  if (!fits(self) || !fits(other) || !fits(out)) {
    context.fail(Error::InvalidArgument);
    return;
  }
  const float* const a = self.const_data_ptr<float>();
  const float* const b = other.const_data_ptr<float>();
  float* const result = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < static_cast<size_t>(out.numel()); ++i) {
    result[i] = a[i] - b[i];
  }
}

} // namespace

extern "C" const Kernel* et_pybindings_test_kernels(size_t* num_kernels) {
  static const Kernel kernels[] = {
      Kernel("aten::add.out", subtract_out),
  };
  *num_kernels = std::size(kernels);
  return kernels;
}
