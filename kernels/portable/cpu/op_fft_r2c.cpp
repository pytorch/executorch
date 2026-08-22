/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace torch::executor::native {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// A complex-to-complex pass has to read a whole line before it can overwrite
// it. Lines up to this length go through a stack buffer, which keeps the buffer
// at 2 KB for double; longer ones ask the runtime for temporary memory. Only
// multi-dimensional transforms reach this path at all: the single-dimension
// case, which is what torch.fft.rfft lowers to, needs no line buffer.
constexpr size_t kStackLineLimit = 128;

// Mirrors ATen's fft_norm_mode (ATen/native/SpectralOpsUtils.h), which is how
// the normalization argument is encoded.
enum class fft_norm_mode {
  none, // No normalization
  by_root_n, // Divide by sqrt(signal_size)
  by_n, // Divide by signal_size
};

template <typename T>
std::optional<T> compute_fct(
    KernelRuntimeContext& ctx,
    const Tensor& t,
    IntArrayRef dim,
    int64_t normalization) {
  constexpr auto one = static_cast<T>(1);
  const auto mode = static_cast<fft_norm_mode>(normalization);
  if (mode == fft_norm_mode::none) {
    return one;
  }
  int64_t n = 1;
  for (auto idx : dim) {
    n *= t.sizes()[idx];
  }
  switch (mode) {
    case fft_norm_mode::none:
      return one;
    case fft_norm_mode::by_n:
      return one / static_cast<T>(n);
    case fft_norm_mode::by_root_n:
      return one / std::sqrt(static_cast<T>(n));
  }
  ET_KERNEL_CHECK_MSG(
      ctx,
      false,
      InvalidArgument,
      std::nullopt,
      "Unsupported normalization type: %" PRId64,
      normalization);
}

// cos and sin of -2*pi*idx/n.
//
// The quarter turns are returned exactly rather than through cos/sin, so that a
// real input's Nyquist bin comes out with a zero imaginary part instead of
// rounding noise on the order of 1e-16. Reducing idx modulo n first also keeps
// the angle inside one period, which matters for accuracy once k * j is large.
void twiddle(size_t idx, size_t n, double& cos_out, double& sin_out) {
  idx %= n;
  if (idx == 0) {
    cos_out = 1.0;
    sin_out = 0.0;
  } else if (2 * idx == n) {
    cos_out = -1.0;
    sin_out = 0.0;
  } else if (4 * idx == n) {
    cos_out = 0.0;
    sin_out = -1.0;
  } else if (4 * idx == 3 * n) {
    cos_out = 0.0;
    sin_out = 1.0;
  } else {
    const double angle =
        -kTwoPi * static_cast<double>(idx) / static_cast<double>(n);
    cos_out = std::cos(angle);
    sin_out = std::sin(angle);
  }
}

// Offset of the start of the line_index'th line along `axis`, for a tensor with
// the given sizes and strides. Lines are enumerated over every dimension except
// `axis`, so the same index names the same logical line in two tensors that
// agree on all dimensions but that one.
size_t line_offset(
    size_t line_index,
    ArrayRef<Tensor::SizesType> sizes,
    ArrayRef<Tensor::StridesType> strides,
    size_t axis) {
  size_t offset = 0;
  for (size_t d = sizes.size(); d-- > 0;) {
    if (d == axis) {
      continue;
    }
    const size_t size = static_cast<size_t>(sizes[d]);
    offset += (line_index % size) * static_cast<size_t>(strides[d]);
    line_index /= size;
  }
  return offset;
}

// Forward real-to-complex DFT along `axis`, writing the onesided output.
// The normalization factor is folded in here: every later pass is linear, so
// applying it once at the front scales the whole transform.
template <typename T>
void dft_r2c_axis(const Tensor& in, Tensor& out, size_t axis, T fct) {
  using C = executorch::runtime::etensor::complex<T>;
  const T* const in_data = in.const_data_ptr<T>();
  C* const out_data = out.mutable_data_ptr<C>();

  const size_t n = static_cast<size_t>(in.size(axis));
  const size_t n_out = static_cast<size_t>(out.size(axis));
  const size_t in_stride = static_cast<size_t>(in.strides()[axis]);
  const size_t out_stride = static_cast<size_t>(out.strides()[axis]);
  const size_t num_lines = n == 0 ? 0 : static_cast<size_t>(in.numel()) / n;

  for (size_t line = 0; line < num_lines; ++line) {
    const size_t in_off = line_offset(line, in.sizes(), in.strides(), axis);
    const size_t out_off = line_offset(line, out.sizes(), out.strides(), axis);
    for (size_t k = 0; k < n_out; ++k) {
      double real = 0;
      double imag = 0;
      for (size_t j = 0; j < n; ++j) {
        double c = 0;
        double s = 0;
        twiddle(k * j, n, c, s);
        const double x = static_cast<double>(in_data[in_off + j * in_stride]);
        real += x * c;
        imag += x * s;
      }
      out_data[out_off + k * out_stride] =
          C{static_cast<T>(real * static_cast<double>(fct)),
            static_cast<T>(imag * static_cast<double>(fct))};
    }
  }
}

// In-place forward complex-to-complex DFT along `axis`. `scratch` must hold at
// least out.size(axis) elements.
template <typename T>
void dft_c2c_axis_(Tensor& out, size_t axis, void* scratch) {
  using C = executorch::runtime::etensor::complex<T>;
  C* const out_data = out.mutable_data_ptr<C>();
  C* const line_buf = static_cast<C*>(scratch);

  const size_t n = static_cast<size_t>(out.size(axis));
  const size_t stride = static_cast<size_t>(out.strides()[axis]);
  const size_t num_lines = n == 0 ? 0 : static_cast<size_t>(out.numel()) / n;

  for (size_t line = 0; line < num_lines; ++line) {
    const size_t off = line_offset(line, out.sizes(), out.strides(), axis);
    for (size_t j = 0; j < n; ++j) {
      line_buf[j] = out_data[off + j * stride];
    }
    for (size_t k = 0; k < n; ++k) {
      double real = 0;
      double imag = 0;
      for (size_t j = 0; j < n; ++j) {
        double c = 0;
        double s = 0;
        twiddle(k * j, n, c, s);
        const double xr = static_cast<double>(line_buf[j].real_);
        const double xi = static_cast<double>(line_buf[j].imag_);
        real += xr * c - xi * s;
        imag += xr * s + xi * c;
      }
      out_data[off + k * stride] =
          C{static_cast<T>(real), static_cast<T>(imag)};
    }
  }
}

} // namespace

// Reference discrete Fourier transform.
//
// This is a direct O(n^2) evaluation of the transform sum, not a fast Fourier
// transform. kernels/optimized provides a pocketfft-backed _fft_r2c.out that is
// asymptotically faster; this exists so that a graph containing _fft_r2c can be
// run by a build that only has the portable kernels, rather than failing to
// load with OperatorMissing. Audio front-ends that transform a few hundred
// points per frame are the intended case.
Tensor& _fft_r2c_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  auto in_sizes = in.sizes();
  ET_KERNEL_CHECK(ctx, in.dim() <= kTensorDimensionLimit, InvalidArgument, out);
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

  std::array<Tensor::SizesType, kTensorDimensionLimit> out_sizes_storage;
  executorch::runtime::Span<Tensor::SizesType> out_sizes(
      out_sizes_storage.data(), in_sizes.size());
  std::copy(in_sizes.begin(), in_sizes.end(), out_sizes.begin());
  out_sizes[dim.back()] = out_sizes[dim.back()] / 2 + 1;

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

  // NOTE: as of this writing, upstream PyTorch only supports float/double, so
  // we follow suit.
  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "_fft_r2c.out", CTYPE_IN, [&] {
    auto fct = compute_fct<CTYPE_IN>(ctx, in, dim, normalization);
    if (!fct) {
      // Check failed, just bail out of the lambda.
      return;
    }

    // The real transform runs along the last requested dimension, which is the
    // one that is halved; the remaining dimensions are complex transforms of
    // the result, matching pocketfft's multi-axis r2c.
    const size_t real_axis = static_cast<size_t>(dim.back());
    dft_r2c_axis<CTYPE_IN>(in, out, real_axis, *fct);

    if (dim.size() == 1) {
      return;
    }

    using Complex = executorch::runtime::etensor::complex<CTYPE_IN>;
    size_t max_line = 0;
    for (size_t i = 0; i + 1 < dim.size(); ++i) {
      max_line = std::max(max_line, static_cast<size_t>(out.size(dim[i])));
    }

    std::array<Complex, kStackLineLimit> stack_buf;
    void* line_buf = stack_buf.data();
    if (max_line > kStackLineLimit) {
      Result<void*> scratch = ctx.allocate_temp(max_line * sizeof(Complex));
      ET_KERNEL_CHECK_MSG(
          ctx,
          scratch.ok(),
          MemoryAllocationFailed,
          ,
          "_fft_r2c needs %zu bytes of temporary memory to transform a "
          "dimension of length %zu, but no temp allocator is available",
          max_line * sizeof(Complex),
          max_line);
      line_buf = scratch.get();
    }

    for (size_t i = 0; i + 1 < dim.size(); ++i) {
      dft_c2c_axis_<CTYPE_IN>(out, static_cast<size_t>(dim[i]), line_buf);
    }
  });

  return out;
}

} // namespace torch::executor::native
