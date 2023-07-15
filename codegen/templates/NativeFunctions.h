// clang-format off
#pragma once

#include <tuple>

#include <executorch/runtime/core/exec_aten/exec_aten.h> // at::Tensor etc.
#include <executorch/codegen/macros.h> // TORCH_API
#include <executorch/runtime/kernel/kernel_runtime_context.h>

// ${generated_comment}

${nativeFunctions_declarations}
