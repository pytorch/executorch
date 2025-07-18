#include <executorch/runtime/kernel/kernel_includes.h>
#include <iostream>

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& softmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
    // Your optimized implementation here
    // Fill 'out' with the result and return it
    std::cout << "xxxxxxxxxx softmax_out kernel called" << std::endl;
    std::cout.flush();
    ET_LOG(Error, "xxxxxxxxxx softmax_out kernel called");

  return out;
}

Tensor softmax(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    bool half_to_float) {
    std::cout << "xxxxxxxxxx softmax_default kernel called" << std::endl;
    std::cout.flush();
    ET_LOG(Error, "xxxxxxxxxx softmax_default kernel called");
    return self;
}

} // namespace native
} // namespace cortex_m
