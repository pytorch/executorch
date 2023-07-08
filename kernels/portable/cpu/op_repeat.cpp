// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/kernels/portable/cpu/util/repeat_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {
namespace {

void calculate_output_size(
    const exec_aten::ArrayRef<exec_aten::SizesType>& self_sizes,
    const exec_aten::ArrayRef<int64_t>& repeats,
    Tensor::SizesType* out_sizes_ptr) {
  ET_CHECK_MSG(
      repeats.size() >= self_sizes.size(),
      "Repeats vector size is %zu must be >= self_sizes %zu.",
      repeats.size(),
      self_sizes.size());
  int32_t i = 0;
  for (; i < (repeats.size() - self_sizes.size()); ++i) {
    out_sizes_ptr[i] = static_cast<exec_aten::SizesType>(repeats[i]);
  }
  int32_t j = 0;
  for (; i < repeats.size(); ++i) {
    out_sizes_ptr[i] =
        static_cast<exec_aten::SizesType>(repeats[i]) * self_sizes[j];
    j++;
  }
}

} // namespace

using Tensor = exec_aten::Tensor;

// repeat.out(Tensor self, int[] repeats, *, Tensor(a!) out) -> Tensor(a!)
Tensor& repeat_out(
    RuntimeContext& context,
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> repeats,
    Tensor& out) {
  (void)context;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  calculate_output_size(self.sizes(), repeats, expected_output_size);
  auto error = resize_tensor(out, {expected_output_size, repeats.size()});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  return repeat_tensor(self, repeats, out);
}

} // namespace native
} // namespace executor
} // namespace torch
