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

torch::executor::Tensor& aten_softmax(
    torch::executor::KernelRuntimeContext& context,
    const torch::executor::Tensor& self,
    int64_t dim,
    bool half_to_float,
    torch::executor::Tensor& out) {
  // Your CMSIS-NN optimized implementation here
  // Return 'out' tensor as per Executorch kernel signature
  //std::cout << "softmax kernel called" << std::endl;
  ET_LOG(Info, "xxxxxxxxxx softmax kernel called");

  //assert(false);
  //assert(true);
  return out;
}

} // namespace native
} // namespace cortex_m
