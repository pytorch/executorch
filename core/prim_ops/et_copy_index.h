#pragma once

#include <executorch/core/values/Evalue.h>
#include <executorch/kernels/kernel_runtime_context.h>

namespace torch {
namespace executor {
namespace function {

void et_copy_index(RuntimeContext& context, EValue** stack);

} // namespace function
} // namespace executor
} // namespace torch
