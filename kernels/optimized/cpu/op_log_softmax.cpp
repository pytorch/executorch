/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <cmath>
#include <type_traits>

#include <ATen/native/cpu/LogSoftmaxKernelImpl.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

// `_log_softmax_out` Applies the Log_Softmax function to an n-dimensional input
// Tensor rescaling them so that the elements of the n-dimensional output
// Tensor.

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
namespace {

template <typename IN_T, typename OUT_T>
void log_softmax_kernel(const Tensor& input, int64_t dim, Tensor& out) {
  const IN_T* ET_RESTRICT input_data_base = input.const_data_ptr<IN_T>();
  OUT_T* ET_RESTRICT output_data_base = out.mutable_data_ptr<OUT_T>();

  if (input.dim() == 0) {
    output_data_base[0] = 0;
    return;
  }

  int64_t dim_size = input.size(dim);

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer_size *= input.size(i);
  }
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    inner_size *= input.size(i);
  }

  if (dim == input.dim() - 1) {
    ::executorch::extension::parallel_for(
        0,
        outer_size,
        ::executorch::extension::internal::GRAIN_SIZE,
        [&](const auto begin, const auto end) {
          at::native::serial_vec_log_softmax_lastdim_range(
              input_data_base,
              output_data_base,
              dim_size,
              at::native::vec_log_softmax_lastdim_chunk_size<IN_T>(
                  executorch::extension::internal::GRAIN_SIZE,
                  outer_size,
                  dim_size),
              begin,
              end);
        });
  } else {
    // BLOCK_SIZE in PyTorch is intended for server CPUs; let's
    // halve it to try and have a better chance of fitting in mobile
    // chip caches.
    const auto [chunk_size_binding, num_chunks_binding] =
        at::native::vec_logsoftmax_chunk_size_and_num_chunks<
            float,
            /*BLOCK_SIZE=*/64 * 1024>(inner_size, dim_size);
    // Work around "capturing a structured binding is not yet supported in
    // OpenMP".
    const auto chunk_size = chunk_size_binding;
    const auto num_chunks = num_chunks_binding;
    ::executorch::extension::parallel_for(
        0,
        outer_size * num_chunks,
        ::executorch::extension::internal::GRAIN_SIZE,
        [&](const auto begin, const auto end) {
          at::native::serial_vec_logsoftmax_range(
              input_data_base,
              output_data_base,
              inner_size,
              chunk_size,
              num_chunks,
              dim_size,
              begin,
              end);
        });
  }
  return;
}

// OUT_T is the corresponding C++ type for out.scalar_type(). Only takes float
// or double.
template <
    typename OUT_T,
    std::enable_if_t<std::is_floating_point<OUT_T>::value, bool> = true>
bool log_softmax_wrapper(const Tensor& X, int64_t dim, Tensor& out) {
  auto input_scalar_type = X.scalar_type();
  switch (input_scalar_type) {
    // TODO: support Double as well
    case ScalarType::Float:
      log_softmax_kernel<float, OUT_T>(X, dim, out);
      return true;
    default:
      return false; // Unsupported input dtype
  }
}
} // namespace

// _log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)
// -> Tensor(a!)
Tensor& opt_log_softmax_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      check_log_softmax_args(self, dim, half_to_float, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  dim = dim < 0 ? dim + nonzero_dim(self) : dim;

  auto out_scalar_type = out.scalar_type();
  switch (out_scalar_type) {
    // TODO: support Double as well
    case ScalarType::Float: {
      bool success = log_softmax_wrapper<float>(self, dim, out);
      ET_KERNEL_CHECK(context, success, InvalidArgument, out);
      break;
    }
    default:
      ET_KERNEL_CHECK(context, false, InvalidArgument, out);
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
