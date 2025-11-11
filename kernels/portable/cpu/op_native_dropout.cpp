/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <random>
#include <tuple>

namespace torch::executor::native {
std::tuple<Tensor&, Tensor&> native_dropout_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double prob,
    torch::executor::optional<bool> train,
    Tensor& out,
    Tensor& mask) {
  std::tuple<Tensor&, Tensor&> ret(out, mask);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(input, out), InvalidArgument, ret);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(input, out, mask), InvalidArgument, ret);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mask, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret);
  ET_KERNEL_CHECK(ctx, tensor_is_bool_type(mask), InvalidArgument, ret);
  ET_KERNEL_CHECK_MSG(
      ctx,
      prob >= 0 && prob <= 1,
      InvalidArgument,
      ret,
      "dropout probability has to be between 0 and 1 but got %f",
      prob);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "native_dropout.out";
  if ((!train.has_value() || train.value()) && prob != 0) {
    {
      std::mt19937 gen((std::random_device())());
      std::uniform_real_distribution<double> dist;
      bool* const mask_data_ptr = mask.mutable_data_ptr<bool>();
      for (const auto ii : c10::irange(mask.numel())) {
        mask_data_ptr[ii] = dist(gen) >= prob;
      }
    }
    ET_SWITCH_FLOATHBF16_TYPES(
        input.scalar_type(), ctx, op_name, CTYPE_COMPUTE, [&]() {
          utils::apply_bitensor_elementwise_fn<
              CTYPE_COMPUTE,
              op_name,
              utils::SupportedTensorDtypes::SAME_AS_COMMON>(
              [](const CTYPE_COMPUTE val, const CTYPE_COMPUTE mask_val) {
                if (!mask_val) {
                  return static_cast<decltype(val)>(0);
                }
                return val;
              },
              ctx,
              input,
              utils::SupportedTensorDtypes::FLOATHBF16,
              mask,
              // TODO: should really be just BOOL
              utils::SupportedTensorDtypes::BOOL_OR_BYTE,
              out);
        });
  } else if (input.numel() > 0) {
    std::memcpy(out.mutable_data_ptr(), input.data_ptr(), input.nbytes());
    std::memset(mask.mutable_data_ptr(), true, mask.nbytes());
  }
  return ret;
}

} // namespace torch::executor::native
