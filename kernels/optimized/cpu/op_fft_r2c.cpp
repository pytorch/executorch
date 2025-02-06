/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/span.h>
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
pocketfft::stride_t stride_from_tensor(const Tensor& t) {
  pocketfft::stride_t stride(t.strides().begin(), t.strides().end());
  for (auto& s : stride) {
    s *= t.element_size();
  }
  return stride;
}

pocketfft::shape_t shape_from_tensor(const Tensor& t) {
  return pocketfft::shape_t(t.sizes().begin(), t.sizes().end());
}

// NOTE: The reinterpret_cast in tensor_cdata is UB, but it's what
// PyTorch core does and I'm not aware of a portable way to do this
// that doesn't rely on UB.
template <typename T>
inline std::complex<T>* tensor_cdata(Tensor& t) {
  return reinterpret_cast<std::complex<T>*>(
      t.data_ptr<executorch::runtime::etensor::complex<T>>());
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

Tensor& opt_fft_r2c_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  auto in_sizes = in.sizes();
  ET_KERNEL_CHECK(ctx, in.dim() <= kTensorDimensionLimit, InvalidArgument, out);

  std::array<Tensor::SizesType, kTensorDimensionLimit> out_sizes_storage;
  executorch::runtime::Span<Tensor::SizesType> out_sizes(
      out_sizes_storage.data(), in_sizes.size());
  std::copy(in_sizes.begin(), in_sizes.end(), out_sizes.begin());
  ET_KERNEL_CHECK(ctx, !dim.empty(), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      onesided,
      InvalidArgument,
      out,
      "onesided=False is not supported yet in _fft_r2c");

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == executorch::runtime::toComplexType(in.scalar_type()),
      InvalidArgument,
      out,
      "the output type for _fft_r2c must be the Complex type corresponding to the input type");

  for (auto d : dim) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        d >= 0 && d < in.dim(),
        InvalidArgument,
        out,
        "dims must be in bounds (got %" PRId64 ")",
        d);
  }

  if (onesided) {
    out_sizes[dim.back()] = out_sizes[dim.back()] / 2 + 1;
  }
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out,
          executorch::runtime::ArrayRef<Tensor::SizesType>(
              out_sizes.data(), out_sizes.size())) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor (last dim %d).",
      out_sizes[dim.back()]);

  pocketfft::shape_t axes(dim.begin(), dim.end());
  auto in_shape = shape_from_tensor(in);
  // TODO: if arbitrary strides are a possibility, we need to validate
  // these, because pocketfft README says "Strides that lead to
  // multiple accesses of the same memory address are not allowed."
  auto in_stride = stride_from_tensor(in);
  auto out_stride = stride_from_tensor(out);
  // NOTE: as of this writing, upstream PyTorch only supports
  // float/double, so we follow suit.
  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "_fft_r2c.out", CTYPE_IN, [&] {
    auto fct = compute_fct<CTYPE_IN>(ctx, in, dim, normalization);
    if (!fct) {
      // Check failed, just bail out of the lambda.
      return;
    }
    pocketfft::r2c<CTYPE_IN>(
        in_shape,
        in_stride,
        out_stride,
        axes,
        true,
        in.const_data_ptr<CTYPE_IN>(),
        tensor_cdata<CTYPE_IN>(out),
        *fct);

    // TODO: fill with conjugate symmetry if not onesided; see
    // ATen/native/mkl/SpectralOps.cpp
  });
  return out;
}
} // namespace torch::executor::native
