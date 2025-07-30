#include <executorch/runtime/kernel/kernel_includes.h>
#include <iostream>

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& aten_add_tensor(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& other,
    const ScalarType dtype,
    Tensor& out) {
  ET_LOG(Info, "xxxxxxxxxx aten_add_tensor kernel called");

  // Ensure input is char type
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Char,
      "self.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(self.scalar_type()));

  ET_CHECK_MSG(
      other.scalar_type() == ScalarType::Char,
      "other.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(other.scalar_type()));

  // Check dtype is int8 (Char)
  ET_CHECK_MSG(
      dtype == ScalarType::Char,
      "dtype %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(dtype));
  
  // Example: element-wise add self and other into out
  // (Assuming Tensor has data() and size() methods)
 const int8_t* self_data = self.const_data_ptr<int8_t>();
  const int8_t* other_data = other.const_data_ptr<int8_t>();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();
  size_t numel = self.numel(); // or self.size() if that's the API
  for (size_t i = 0; i < numel; ++i) {
    out_data[i] = self_data[i] + other_data[i];
  }
  return out;
}

} // namespace native
} // namespace cortex_m
