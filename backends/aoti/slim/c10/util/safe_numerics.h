#pragma once

// Thin wrapper to reuse ExecuTorch's c10 safe_numerics implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::{safe_multiplies_u64, add_overflows,
// mul_overflows}.
//
// NOTE: multiply_integers is defined in accumulate.h (SlimTensor-specific).
// NOTE: sub_overflows is not available in ET c10 safe_numerics.

#include <c10/util/safe_numerics.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {

using ::c10::add_overflows;
using ::c10::mul_overflows;
using ::c10::safe_multiplies_u64;

} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch
