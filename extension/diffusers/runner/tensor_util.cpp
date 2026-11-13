/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/tensor_util.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>

namespace executorch::extension::diffusers {

using executorch::aten::BFloat16;
using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

template <typename T>
std::vector<float> widen(const T* data, size_t numel) {
  std::vector<float> values(numel);
  for (size_t i = 0; i < numel; ++i) {
    values[i] = static_cast<float>(data[i]);
  }
  return values;
}

} // namespace

Result<HostTensor> to_host(const executorch::aten::Tensor& tensor) {
  HostTensor host;
  host.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
  const size_t numel = tensor.numel();
  switch (tensor.scalar_type()) {
    case ScalarType::Float:
      host.data = widen(tensor.const_data_ptr<float>(), numel);
      break;
    case ScalarType::Half:
      host.data = widen(tensor.const_data_ptr<Half>(), numel);
      break;
    case ScalarType::BFloat16:
      host.data = widen(tensor.const_data_ptr<BFloat16>(), numel);
      break;
    default:
      ET_LOG(
          Error,
          "Expected a floating point tensor, got dtype %d",
          static_cast<int>(tensor.scalar_type()));
      return Error::InvalidArgument;
  }
  return host;
}

TensorPtr to_tensor(const HostTensor& host, ScalarType dtype) {
  return make_tensor_ptr(host.sizes, host.data, {}, {}, dtype);
}

void round_to(std::vector<float>& values, ScalarType dtype) {
  if (dtype == ScalarType::Half) {
    std::transform(
        values.begin(), values.end(), values.begin(), [](float value) {
          return static_cast<float>(static_cast<Half>(value));
        });
  } else if (dtype == ScalarType::BFloat16) {
    std::transform(
        values.begin(), values.end(), values.begin(), [](float value) {
          return static_cast<float>(static_cast<BFloat16>(value));
        });
  }
}

std::vector<float>
randn(TorchGenerator& generator, ScalarType dtype, size_t numel) {
  switch (dtype) {
    case ScalarType::Half: {
      const auto values = generator.randn<Half>(numel);
      return widen(values.data(), numel);
    }
    case ScalarType::BFloat16: {
      const auto values = generator.randn<BFloat16>(numel);
      return widen(values.data(), numel);
    }
    default:
      return generator.randn<float>(numel);
  }
}

} // namespace executorch::extension::diffusers
