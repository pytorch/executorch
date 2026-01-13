/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>
#include <pocketfft_hdronly.h>
#include <optional>

namespace torch::executor::native {

// TODO: contents of this anonymous namespace are copy/pasted from
// PyTorch core (aten/src/ATen/native/mkl/SpectralOps.cpp). Small
// portions (the parts that don't depend on Tensor) could be reused;
// refactor to enable that once we can share headers from PyTorch
// core.
namespace {
inline pocketfft::stride_t stride_from_tensor(const Tensor& t) {
  pocketfft::stride_t stride(t.strides().begin(), t.strides().end());
  for (auto& s : stride) {
    s *= t.element_size();
  }
  return stride;
}

inline pocketfft::shape_t shape_from_tensor(const Tensor& t) {
  return pocketfft::shape_t(t.sizes().begin(), t.sizes().end());
}

// NOTE: The reinterpret_cast in tensor_cdata is UB, but it's what
// PyTorch core does and I'm not aware of a portable way to do this
// that doesn't rely on UB.
template <typename T>
inline std::complex<T>* tensor_cdata(Tensor& t) {
  return reinterpret_cast<std::complex<T>*>(
      t.mutable_data_ptr<executorch::runtime::etensor::complex<T>>());
}

template <typename T>
inline const std::complex<T>* tensor_cdata(const Tensor& t) {
  return reinterpret_cast<const std::complex<T>*>(
      t.const_data_ptr<executorch::runtime::etensor::complex<T>>());
}

// NOTE: in particular this is in ATen/native/SpectralOpsUtils.h and
// could be shared immediately.
enum class fft_norm_mode {
  none, // No normalization
  by_root_n, // Divide by sqrt(signal_size)
  by_n, // Divide by signal_size
};

// NOTE: slight fork from upstream PyTorch to use ET_KERNEL_CHECK;
// upstream with TORCH_CHECK will be fine to use once we have code
// sharing.
template <typename T>
std::optional<T>
compute_fct(KernelRuntimeContext& ctx, int64_t size, int64_t normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (static_cast<fft_norm_mode>(normalization)) {
    case fft_norm_mode::none:
      return one;
    case fft_norm_mode::by_n:
      return one / static_cast<T>(size);
    case fft_norm_mode::by_root_n:
      return one / std::sqrt(static_cast<T>(size));
  }
  ET_KERNEL_CHECK_MSG(
      ctx,
      false,
      InvalidArgument,
      std::nullopt,
      "Unsupported normalization type: %" PRId64,
      normalization);
}

template <typename T>
std::optional<T> compute_fct(
    KernelRuntimeContext& ctx,
    const Tensor& t,
    IntArrayRef dim,
    int64_t normalization) {
  if (static_cast<fft_norm_mode>(normalization) == fft_norm_mode::none) {
    return static_cast<T>(1);
  }
  const auto& sizes = t.sizes();
  int64_t n = 1;
  for (auto idx : dim) {
    n *= sizes[idx];
  }
  return compute_fct<T>(ctx, n, normalization);
}
} // namespace

} // namespace torch::executor::native
