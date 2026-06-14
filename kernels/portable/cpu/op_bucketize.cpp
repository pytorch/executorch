#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

namespace torch {
namespace executor {
namespace native {
namespace {

using executorch::runtime::isRealHBF16Type;
using torch::executor::native::utils::SupportedTensorDtypes;
using torch::executor::native::utils::internal::get_load_to_compute_fn;
using torch::executor::native::utils::internal::load_to_compute_fn;

constexpr int64_t BUCKETIZE_GRAIN_SIZE = 200;

template <typename CTYPE>
int64_t cus_lower_bound(
    int64_t end,
    const CTYPE val,
    const char* bd,
    load_to_compute_fn<CTYPE> bd_load_fn,
    ssize_t bd_elem_size) {
  int64_t start = 0;

  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    CTYPE mid_bd = bd_load_fn(&bd[mid * bd_elem_size]);

    if (mid_bd < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename CTYPE>
int64_t cus_upper_bound(
    int64_t end,
    const CTYPE val,
    const char* bd,
    load_to_compute_fn<CTYPE> bd_load_fn,
    ssize_t bd_elem_size) {
  int64_t start = 0;

  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    CTYPE mid_bd = bd_load_fn(&bd[mid * bd_elem_size]);

    if (mid_bd <= val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename CTYPE_COMPUTE, typename CTYPE_OUT, const char* op_name>
void bucketize_tensor_impl(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& boundaries,
    const bool& right,
    Tensor& out) {
  auto in_load_fn = get_load_to_compute_fn<CTYPE_COMPUTE, op_name>(
      context, self, SupportedTensorDtypes::REALHBF16);
  const ssize_t in_size = self.element_size();
  auto in_data = reinterpret_cast<const char*>(self.const_data_ptr());

  auto bd_load_fn = get_load_to_compute_fn<CTYPE_COMPUTE, op_name>(
      context, boundaries, SupportedTensorDtypes::REALHBF16);
  const ssize_t bd_elem_size = boundaries.element_size();
  auto bd_data = reinterpret_cast<const char*>(boundaries.const_data_ptr());
  int64_t bd_end = boundaries.sizes().back();

  auto out_data = out.mutable_data_ptr<CTYPE_OUT>();

  const bool success = parallel_for(
      0,
      self.numel(),
      BUCKETIZE_GRAIN_SIZE,
      [&](const auto begin, const auto end) {
        for (const auto i : c10::irange(begin, end)) {
          auto compute_val = in_load_fn(&in_data[i * in_size]);
          int64_t pos = right
              ? cus_upper_bound(
                    bd_end, compute_val, bd_data, bd_load_fn, bd_elem_size)
              : cus_lower_bound(
                    bd_end, compute_val, bd_data, bd_load_fn, bd_elem_size);
          out_data[i] = pos;
        }
      });

  ET_KERNEL_CHECK_MSG(context, success, Internal, , "parallel_for failed");
}

template <typename CTYPE_COMPUTE, typename CTYPE_OUT, const char* op_name>
void bucketize_scalar_impl(
    KernelRuntimeContext& context,
    const Scalar self,
    const Tensor& boundaries,
    const bool& right,
    Tensor& out) {
  CTYPE_COMPUTE compute_val = utils::scalar_to<CTYPE_COMPUTE>(self);

  auto bd_load_fn = get_load_to_compute_fn<CTYPE_COMPUTE, op_name>(
      context, boundaries, SupportedTensorDtypes::REALHBF16);
  const ssize_t bd_elem_size = boundaries.element_size();
  auto bd_data = reinterpret_cast<const char*>(boundaries.const_data_ptr());
  int64_t bd_end = boundaries.sizes().back();

  auto out_data = out.mutable_data_ptr<CTYPE_OUT>();

  int64_t pos = right
      ? cus_upper_bound(bd_end, compute_val, bd_data, bd_load_fn, bd_elem_size)
      : cus_lower_bound(bd_end, compute_val, bd_data, bd_load_fn, bd_elem_size);
  out_data[0] = pos;
}

Error bucketize_common_pre_checks(
    const Tensor& boundaries,
    bool out_int32,
    Tensor& out) {
  ET_CHECK_OR_RETURN_ERROR(
      boundaries.dim() == 1,
      InvalidArgument,
      "boundaries tensor must be 1 dimension, but got dim(%zu)",
      boundaries.dim());

  ScalarType out_dtype = out.scalar_type();
  ET_CHECK_OR_RETURN_ERROR(
      (out_dtype == ScalarType::Long && !out_int32) ||
          (out_dtype == ScalarType::Int && out_int32),
      InvalidArgument,
      "torch.bucketize(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on whether out_int32 flag is True, but we got output tensor dtype %s and out_int32 flag is %s",
      toString(out_dtype),
      (out_int32 ? "True" : "False"));

  ScalarType bound_dtype = boundaries.scalar_type();
  ET_CHECK_OR_RETURN_ERROR(
      isRealHBF16Type(bound_dtype),
      InvalidArgument,
      "boundaries tensor of type %s is not supported",
      toString(bound_dtype));

  return Error::Ok;
}

} // namespace

using executorch::runtime::tensor_is_realhbf16_type;

Tensor& bucketize_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& out) {
  ET_KERNEL_CHECK(
      context,
      bucketize_common_pre_checks(boundaries, out_int32, out) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      context, tensors_have_same_shape(self, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      context, tensor_is_realhbf16_type(self), InvalidArgument, out);

  ScalarType common_type =
      promoteTypes(self.scalar_type(), boundaries.scalar_type());
  ScalarType compute_type = utils::get_compute_type(common_type);

  static constexpr const char op_name[] = "bucketize.Tensor_out";

  ET_SWITCH_REALHBF16_TYPES(
      compute_type, context, op_name, CTYPE_COMPUTE, [&]() {
        if (out_int32) {
          bucketize_tensor_impl<CTYPE_COMPUTE, int32_t, op_name>(
              context, self, boundaries, right, out);
        } else {
          bucketize_tensor_impl<CTYPE_COMPUTE, int64_t, op_name>(
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
  ET_KERNEL_CHECK(
      context,
      bucketize_common_pre_checks(boundaries, out_int32, out) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(context, out.dim() == 0, InvalidArgument, out);

  ScalarType common_type =
      utils::promote_type_with_scalar(boundaries.scalar_type(), self);
  ScalarType compute_type = utils::get_compute_type(common_type);

  static constexpr const char op_name[] = "bucketize.Scalar_out";

  ET_SWITCH_REALHBF16_TYPES(
      compute_type, context, op_name, CTYPE_COMPUTE, [&]() {
        if (out_int32) {
          bucketize_scalar_impl<CTYPE_COMPUTE, int32_t, op_name>(
              context, self, boundaries, right, out);
        } else {
          bucketize_scalar_impl<CTYPE_COMPUTE, int64_t, op_name>(
              context, self, boundaries, right, out);
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch