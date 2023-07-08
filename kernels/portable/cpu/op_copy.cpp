// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

// copy.out(const Tensor& self, const Tensor& src, bool non_blocking, Tensor(a!)
// out) -> Tensor(a!), see caffe2/aten/src/ATen/native/Copy.cpp
// TODO: We actually shouldn't see this op with the proper functionalization,
// and this op needs to be deleted
Tensor& copy_out(
    RuntimeContext& context,
    const Tensor& self,
    const Tensor& src,
    bool non_blocking,
    Tensor& out) {
  (void)context;
  // Right now we only support blocking data transfer
  ET_CHECK(non_blocking == false);

  // The srs and out shall share same dtype, but not necessarily for self,
  // because `auto intermediate = src.to(self, non_blocking)` doesn't restrict
  // the type of self. In this kernel we didn't do `to` inside the op. If
  // in the short term we need self in a different type, can extend the op to
  // cover it.
  ET_CHECK_SAME_DTYPE3(self, src, out);

  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_output_dim = 0;

  ET_CHECK_MSG(
      tensor_is_broadcastable_to(src, self),
      "can't broadcast from self to src");
  get_broadcast_target_size(
      self,
      src,
      expected_output_size,
      kTensorDimensionLimit,
      &expected_output_dim);

  ET_CHECK_MSG(
      Error::Ok ==
          resize_tensor(out, {expected_output_size, expected_output_dim}),
      "Failed to resize output tensor.");
  bool to_be_broadcasted_src = !out.sizes().equals(src.sizes());

  //   ET_CHECK_SAME_SHAPE2(expected_output_size, self);
  const Tensor& broadcasted_src =
      to_be_broadcasted_src ? broadcast_tensor(src, out) : src;

  if (broadcasted_src.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(
        out.data_ptr(), broadcasted_src.data_ptr(), broadcasted_src.nbytes());
  }
  if (to_be_broadcasted_src) {
    free_broadcast_tensor(broadcasted_src);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
