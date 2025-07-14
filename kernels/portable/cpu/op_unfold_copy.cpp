#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cstring>
namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

// unfold_copy(Tensor self, int dimension, int size, int step, *, Tensor(a!)
// out) -> Tensor(a!)
Tensor& unfold_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    int64_t size,
    int64_t step,
    Tensor& out) {
  (void)ctx;
  // Check if dimension is valid
  ET_KERNEL_CHECK(
      ctx, check_unfold_copy_args(self, dim, size, step), InvalidArgument, out);
  if (dim < 0) {
    dim += nonzero_dim(self);
  }
  // Calculate output size
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;

  get_unfold_copy_out_target_size(
      self, dim, size, step, expected_output_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_output_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  // Copy data
  const size_t leading_dims = getLeadingDims(self, dim);
  const size_t trailing_dims = getTrailingDims(self, dim);
  ScalarType in_type = self.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "unfold_copy.out", CTYPE_IN, [&]() {
    const CTYPE_IN* input_ptr = self.const_data_ptr<CTYPE_IN>();
    ET_SWITCH_REALHBBF16_TYPES(
        out_type, ctx, "unfold_copy.out", CTYPE_OUT, [&] {
          CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
          for (const auto i : c10::irange(leading_dims)) {
            const CTYPE_IN* src =
                input_ptr + i * self.size(dim) * trailing_dims;
            for (const auto j : c10::irange(out.size(dim))) {
              const CTYPE_IN* dim_src = src + j * step * trailing_dims;
              for (const auto k : c10::irange(trailing_dims)) {
                for (const auto l : c10::irange(size)) {
                  *out_ptr = convert<CTYPE_OUT, CTYPE_IN>(
                      dim_src[k + l * trailing_dims]);
                  out_ptr++;
                }
              }
            }
          }
        });
  });
  return out;
}
} // namespace native
} // namespace executor
} // namespace torch
