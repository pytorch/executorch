// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>

/**
 * torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) → Tensor
 * Performs a matrix multiplication of the matrices mat1 and mat2. The matrix
 * input is added to the final result.
 *
 * If mat1 is a (n \times m)(n×m) tensor, mat2 is a (m \times p)(m×p) tensor,
 * then input must be broadcastable with a (n \times p)(n×p) tensor and out will
 * be a (n \times p)(n×p) tensor.
 *
 * alpha and beta are scaling factors on matrix-vector product between mat1 and
 * mat2 and the added matrix input respectively.
 *
 * out= β input+α (mat1 @ mat2)
 * If beta is 0, then input will be ignored, and nan and inf in it will not be
 * propagated.
 *
 * For inputs of type FloatTensor or DoubleTensor, arguments beta and alpha must
 * be real numbers, otherwise they should be integers.
 */
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;

namespace {

/**
 * Asserts that the parameters are valid.
 * mat1 (m x n), mat2 (n x p), out (m, p), self (m x p)
 * z[i][j] = sum(x[i][k] * y[k][j]), for k in range(n)
 */
void check_addmm_out_args(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  // Ensure self can be broadcasted to out
  ET_CHECK_MSG(
      tensor_is_broadcastable_to(self, out),
      "input tensor can not be broadcasted to out");
  // Ensure dimension is 2 for all tensors.
  // Does not test self here because it will be broadcasted to out.size() after
  // this function, so we just need to ensure out.dim() meets the requirement.
  ET_CHECK_MSG(mat1.dim() == 2, "mat1.dim() %zd != 2", mat1.dim());
  ET_CHECK_MSG(mat2.dim() == 2, "mat2.dim() %zd != 2", mat2.dim());
  ET_CHECK_MSG(out.dim() == 2, "out.dim() %zd != 2", out.dim());
  // Ensure 4 tensors are having the same dtype
  ET_CHECK_SAME_DTYPE3(self, mat1, mat2);
  ET_CHECK_SAME_DTYPE2(self, out);
  // Ensure beta and alpha are having the same type. Maybe support mixing types
  // in the future
  ET_CHECK_SCALAR_SAME_TYPE(beta, alpha);
  // Ensure the out size is compatible with input tensors
  ET_CHECK_MSG(
      mat2.size(1) == out.size(1),
      "mat2.size(1) %zd != out.size(1) %zd",
      mat2.size(1),
      out.size(1));
  ET_CHECK_MSG(
      mat1.size(0) == out.size(0),
      "mat1.size(0) %zd != out.size(0) %zd",
      mat1.size(0),
      out.size(0));
  // Ensure mat1 is able to multiply with mat2
  ET_CHECK_MSG(
      mat1.size(1) == mat2.size(0),
      "mat1.size(1) %zd != mat2.size(0) %zd",
      mat1.size(1),
      mat2.size(0));
}

// for simplicity, assuming all tensors are of the same type and all scalars are
// the same type. `self` can be broadasted to mat1@mat2. T is the tensor dtype
// and we are handling scalar types inside.
template <typename T>
Tensor& addmm_out_kernel(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  const T* __restrict__ self_data = self.data_ptr<T>();
  const T* __restrict__ mat1_data = mat1.data_ptr<T>();
  const T* __restrict__ mat2_data = mat2.data_ptr<T>();
  T* __restrict__ out_data = out.data_ptr<T>();

  size_t m = mat1.size(0);
  size_t n = mat1.size(1);
  size_t p = mat2.size(1);

  if (beta.isBoolean()) {
    vec_addmm<T, bool>(
        out_data,
        self_data,
        mat1_data,
        mat2_data,
        m,
        n,
        p,
        beta.to<bool>(),
        alpha.to<bool>());
  } else if (beta.isIntegral(/*includeBool=*/false)) {
    vec_addmm<T, int64_t>(
        out_data,
        self_data,
        mat1_data,
        mat2_data,
        m,
        n,
        p,
        beta.to<int64_t>(),
        alpha.to<int64_t>());
  } else if (beta.isFloatingPoint()) {
    vec_addmm<T, double>(
        out_data,
        self_data,
        mat1_data,
        mat2_data,
        m,
        n,
        p,
        beta.to<double>(),
        alpha.to<double>());
  } else {
    ET_CHECK_MSG(false, "Unhandled scalar type");
  }
  return out;
}

void resize_out_tensor(const Tensor& mat1, const Tensor& mat2, Tensor& out) {
  Tensor::SizesType expected_output_size[2];
  expected_output_size[0] = mat1.size(0);
  expected_output_size[1] = mat2.size(1);

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in addmm_out");
}
} // namespace

/**
 * addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar
 * alpha=1, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& addmm_out(
    RuntimeContext& context,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  resize_out_tensor(mat1, mat2, out);
  check_addmm_out_args(self, mat1, mat2, beta, alpha, out);

  // The tensor self needs to be broadcasted iff its is size differnet from the
  // target one (out.size())
  bool broadcasted = !out.sizes().equals(self.sizes());
  const Tensor& broadcasted_tensor =
      broadcasted ? broadcast_tensor(self, out) : self;
  auto scalar_type = broadcasted_tensor.scalar_type();

#define ADDMM_TENSOR(ctype, dtype)                                             \
  case ScalarType::dtype:                                                      \
    addmm_out_kernel<ctype>(broadcasted_tensor, mat1, mat2, beta, alpha, out); \
    break;

  switch (scalar_type) {
    ET_FORALL_REAL_TYPES(ADDMM_TENSOR)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", scalar_type);
  }
#undef ADDMM_TENSOR

  if (broadcasted) {
    free_broadcast_tensor(broadcasted_tensor);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
