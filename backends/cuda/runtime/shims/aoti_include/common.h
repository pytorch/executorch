#pragma once

#include <array>
#include <filesystem>
#include <optional>

#include <executorch/backends/cuda/runtime/shims/aoti_runtime/interface.h>
#include <executorch/backends/cuda/runtime/shims/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <executorch/backends/cuda/runtime/shims/aoti_runtime/scalar_to_tensor.h>

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
