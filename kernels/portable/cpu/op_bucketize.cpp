#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <cstdio>
#include <iostream>

namespace torch {
namespace executor {
namespace native {

namespace {

template <typename CTYPE>
int64_t
cus_lower_bound(int64_t start, int64_t end, const CTYPE val, const CTYPE* bd) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    if (bd[mid] < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename CTYPE>
int64_t
cus_upper_bound(int64_t start, int64_t end, const CTYPE val, const CTYPE* bd) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    if (bd[mid] <= val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename CTYPE, typename OUT_CTYPE>
void searchsorted_cpu(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& boundaries,
    const bool& right,
    Tensor& out) {
  const auto bd_data = boundaries.const_data_ptr<CTYPE>();
  const auto in_data = input.const_data_ptr<CTYPE>();
  OUT_CTYPE* out_data = out.mutable_data_ptr<OUT_CTYPE>();
  int64_t end_bd = boundaries.sizes().back();

  const bool success = parallel_for(
      0, input.numel(), 200, [&](const auto begin, const auto end) {
        for (const auto out_i : c10::irange(begin, end)) {
          int64_t pos = right
              ? cus_upper_bound(0, end_bd, in_data[out_i], bd_data)
              : cus_lower_bound(0, end_bd, in_data[out_i], bd_data);
          out_data[out_i] = pos;
        }
      });
  ET_KERNEL_CHECK_MSG(context, success, Internal, , "parallel_for failed");
}

void bucketize_pre_check(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& boundaries,
    bool out_int32,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      context,
      boundaries.dim() == 1,
      InvalidArgument,
      ,
      "boundaries tensor must be 1 dimension, but got dim(",
      boundaries.dim(),
      ")");

  ScalarType out_dtype = out.scalar_type();
  ET_KERNEL_CHECK_MSG(
      context,
      (out_dtype == ScalarType::Long && !out_int32) ||
          (out_dtype == ScalarType::Int && out_int32),
      InvalidArgument,
      ,
      "torch.bucketize(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on ",
      "whether out_int32 flag is True, but we got output tensor's dtype ",
      out_dtype,
      " and out_int32 flag is ",
      (out_int32 ? "True" : "False"));

  ET_KERNEL_CHECK(
      context, tensors_have_same_shape(input, out), InvalidArgument, );
}

} // namespace

Tensor& bucketize_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& out) {
  bucketize_pre_check(context, self, boundaries, out_int32, out);

  ScalarType common_type =
      promoteTypes(self.scalar_type(), boundaries.scalar_type());

  ET_SWITCH_REALHBF16_TYPES(
      common_type, context, "bucketize.Tensor_out", CTYPE, [&]() {
        if (out_int32) {
          searchsorted_cpu<CTYPE, int32_t>(
              context, self, boundaries, right, out);
        } else {
          searchsorted_cpu<CTYPE, int64_t>(
              context, self, boundaries, right, out);
        }
      });
  return out;
}
Tensor& bucketize_scalar_out(
    KernelRuntimeContext& context,
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& out) {
  return out;

} // namespace
} // namespace native
} // namespace executor
} // namespace torch