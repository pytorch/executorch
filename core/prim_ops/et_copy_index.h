#pragma once

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/values/Evalue.h>

namespace torch {
namespace executor {
namespace function {

void et_copy_index(RuntimeContext& context, EValue** stack);

} // namespace function
} // namespace executor
} // namespace torch
