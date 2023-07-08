// clang-format off
#pragma once

#include <tuple>

#include <executorch/core/kernel_types/kernel_types.h> // at::Tensor etc.
#include <executorch/core/macros.h> // TORCH_API
#include <executorch/runtime/kernel/kernel_runtime_context.h>

// ${generated_comment}

${static_dispatch_extra_headers}

namespace torch {
namespace executor {

${Functions_declarations}

} // namespace executor
} // namespace torch
