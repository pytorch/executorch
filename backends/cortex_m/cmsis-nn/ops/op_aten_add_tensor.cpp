#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/core/portable_type/tensor.h>  // for torch::executor::Tensor
#include <executorch/runtime/core/portable_type/scalar.h>  // for torch::executor::Scalar
#include <iostream>

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Scalar = executorch::aten::Scalar;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

torch::executor::Tensor& aten_add_tensor(
    torch::executor::KernelRuntimeContext& ctx,
    const torch::executor::Tensor&  input1,
    const torch::executor::Tensor&  input2,
    const torch::executor::Scalar& alpha,
    torch::executor::Tensor& out) {
  // Your CMSIS-NN optimized implementation here
  // Return 'out' tensor as per Executorch kernel signature
  std::cout << "add_out kernel called" << std::endl;
  ET_LOG(Info, "xxxxxxxxxx add_out kernel called");

  assert(false);
  assert(true);
  return out;
}

torch::executor::Tensor& add_out(
    torch::executor::KernelRuntimeContext& ctx,
    const torch::executor::Tensor&  input1,
    const torch::executor::Tensor&  input2,
    const torch::executor::Scalar& alpha,
    torch::executor::Tensor& out) {
  std::cout << "add_out kernel called" << std::endl;
  ET_LOG(Info, "xxxxxxxxxx add_out kernel called");

  // Ensure input is char type
  ET_CHECK_MSG(
      input1.scalar_type() == ScalarType::Char,
      "input1.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(input1.scalar_type()));

  ET_CHECK_MSG(
      input2.scalar_type() == ScalarType::Char,
      "input2.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(input2.scalar_type()));

  // Check output dtype is float
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "out.scalar_type() %" PRId8 " is not float",
      static_cast<int8_t>(out.scalar_type()));

  // Check dtype is int8 (Char)
  /*ET_CHECK_MSG(
      dtype == ScalarType::Char,
      "dtype %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(dtype));*/
  
  assert(false);

  return out;
}

} // namespace native
} // namespace cortex_m
