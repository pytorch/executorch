// clang-format off
#pragma once

#include <executorch/core/kernel_types/kernel_types.h> // at::Tensor etc.
#include <executorch/core/macros.h> // TORCH_API
#include <tuple>
// ${generated_comment}

${static_dispatch_extra_headers}

namespace torch {
namespace executor {

${Functions_declarations}

} // namespace executor
} // namespace torch
