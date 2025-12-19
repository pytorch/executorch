#pragma once

// Thin wrapper to reuse ExecuTorch's c10::bit_cast implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::bit_cast.

#include <c10/util/bit_cast.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {

using ::c10::bit_cast;

} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch
