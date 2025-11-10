#pragma once

#include <cstdint>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in executorch/backends/cuda/runtime/shims/aoti_torch/c/shim.h. The same rule
// applies to other files under executorch/backends/cuda/runtime/shims/aoti_runtime/.

namespace torch::aot_inductor {

enum ConstantType : uint8_t {
  Unknown = 0,
  Parameter = 1,
  Buffer = 2,
  TensorConstant = 3,
  FoldedConstant = 4,
};

} // namespace torch::aot_inductor
