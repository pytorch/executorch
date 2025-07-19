#include <executorch/runtime/kernel/kernel_includes.h>
#include <iostream>

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& input1,
    const Tensor& input2,
    const ScalarType dtype,
    Tensor& out) {
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
  ET_CHECK_MSG(
      dtype == ScalarType::Char,
      "dtype %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(dtype));
  
  assert(false);

  return out;
}

} // namespace native
} // namespace cortex_m
