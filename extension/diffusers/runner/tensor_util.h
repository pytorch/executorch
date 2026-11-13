/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/diffusers/runner/torch_rng.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

#include <cstddef>
#include <vector>

namespace executorch::extension::diffusers {

/**
 * A float32 copy of a tensor, for the arithmetic between method calls.
 */
struct HostTensor {
  std::vector<executorch::aten::SizesType> sizes;
  std::vector<float> data;
};

/** Copies a float32, float16 or bfloat16 tensor into float32. */
runtime::Result<HostTensor> to_host(const executorch::aten::Tensor& tensor);

/** Creates a tensor of dtype holding the values of host. */
TensorPtr to_tensor(const HostTensor& host, executorch::aten::ScalarType dtype);

/** Rounds values to dtype in place, as storing them in a dtype tensor would. */
void round_to(std::vector<float>& values, executorch::aten::ScalarType dtype);

/** Draws torch.randn(numel, dtype=dtype) values, widened to float32. */
std::vector<float> randn(
    TorchGenerator& generator,
    executorch::aten::ScalarType dtype,
    size_t numel);

/** torch.linspace for float32 or float64. */
template <typename T>
std::vector<T> torch_linspace(T start, T end, int steps) {
  if (steps == 1) {
    return {start};
  }
  const T step = (end - start) / static_cast<T>(steps - 1);
  const int halfway = steps / 2;
  std::vector<T> values(steps);
  for (int i = 0; i < steps; ++i) {
    values[i] = i < halfway ? start + step * static_cast<T>(i)
                            : end - step * static_cast<T>(steps - i - 1);
  }
  return values;
}

} // namespace executorch::extension::diffusers
