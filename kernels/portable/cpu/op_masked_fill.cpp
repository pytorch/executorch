// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using Scalar = exec_aten::Scalar;
namespace {

/**
 * Fills the `out` with values of `self` or `value` based on mask.
 *
 * Assumes that the tensors are contiguous, are the same shape,
 * input and output have the same time and mask is tensor of bools.
 * CTYPE should be the C type (like `float` or `int`) that matches
 * the dtype of the tensors.
 */
template <class CTYPE>
void masked_fill_kernel(
    const Tensor& self,
    const Tensor& mask,
    const Scalar& value,
    Tensor& out) {
  ET_DCHECK(self.numel() == mask.numel() && self.numel() == out.numel());
  CTYPE value_v = 0;
  bool ok = utils::extract_scalar(value, &value_v);
  ET_CHECK_MSG(ok, "Invalid fill value: wrong type or out of range");
  const size_t n = self.numel();
  const auto data_self = self.data_ptr<CTYPE>();
  const auto data_mask = mask.data_ptr<bool>();
  auto data_out = out.data_ptr<CTYPE>();
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = data_mask[i] ? value_v : data_self[i];
  }
}

} // namespace

/**
 * Copies `self` to `out` masking some elemnts with `value`.
 *
 * Asserts that `mask` tensor can be broadcasted to `self`, self and out should
 * have same dtype and size, and mask should be boolean tensor.
 *
 * masked_fill_Scalar_out(Tensor self, Tensor other, *, Scalar alpha=1.0,
 * Tensor(a!) out) -> Tensor(a!)
 */
Tensor& masked_fill_scalar_out(
    RuntimeContext& context,
    const Tensor& self,
    const Tensor& mask,
    const Scalar& value,
    Tensor& out) {
  ET_CHECK_MSG(
      tensor_is_broadcastable_to(mask, self),
      "masked_fill_scalar_out operateor can not broadcast mask to self");

  // The mask needs to be broadcasted iff its size differnet from the target one
  // (self.size())
  bool broadcasted = !self.sizes().equals(mask.sizes());
  const Tensor& broadcast_mask =
      broadcasted ? torch::executor::broadcast_tensor(mask, self) : mask;

  torch::executor::Error err = resize_tensor(out, self.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in masked_fill_scalar_out");

  ET_CHECK_SAME_SHAPE_AND_DTYPE2(self, out);
  ET_CHECK_SAME_SHAPE2(self, broadcast_mask);
  ET_CHECK_MSG(
      broadcast_mask.scalar_type() == ScalarType::Bool, "Unexpected mask type");

#define MASKED_FILL(ctype, dtype)                                \
  case ScalarType::dtype:                                        \
    masked_fill_kernel<ctype>(self, broadcast_mask, value, out); \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, MASKED_FILL)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }

#undef MASKED_FILL
  if (broadcasted) {
    free_broadcast_tensor(broadcast_mask);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
