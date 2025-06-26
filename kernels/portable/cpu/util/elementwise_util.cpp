/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>

namespace torch::executor::native::utils::internal {

template <typename... Args>
inline bool validate_elementwise_fn_inputs_impl(
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    ScalarType compute_type,
    Args... inputs) {
  static_assert(
      (std::is_same_v<Args, std::pair<const Tensor*, SupportedTensorDtypes>> &&
       ...));
  const auto check_input_dtype = [](auto input, auto compute_type) {
    return internal::check_tensor_dtype(
        *input.first, input.second, compute_type);
  };
  ET_KERNEL_CHECK(
      ctx,
      (check_input_dtype(inputs, compute_type) && ...) &&
          internal::check_tensor_dtype(out, out_dtypes, compute_type),
      InvalidArgument,
      false);

  return true;
}

bool validate_elementwise_fn_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    ScalarType compute_type,
    std::pair<const Tensor*, SupportedTensorDtypes> input) {
  return validate_elementwise_fn_inputs_impl(
      ctx,
      out,
      out_dtypes,
      compute_type,
      input);
}

bool validate_elementwise_fn_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    ScalarType compute_type,
    std::pair<const Tensor*, SupportedTensorDtypes> input0,
    std::pair<const Tensor*, SupportedTensorDtypes> input1) {
  return validate_elementwise_fn_inputs_impl(
      ctx,
      out,
      out_dtypes,
      compute_type,
      input0,
      input1);
}

bool validate_elementwise_fn_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    ScalarType compute_type,
    std::pair<const Tensor*, SupportedTensorDtypes> input0,
    std::pair<const Tensor*, SupportedTensorDtypes> input1,
    std::pair<const Tensor*, SupportedTensorDtypes> input2) {
  return validate_elementwise_fn_inputs_impl(
      ctx,
      out,
      out_dtypes,
      compute_type,
      input0,
      input1,
      input2);
}


} // namespace torch::executor::native::utils::internal
