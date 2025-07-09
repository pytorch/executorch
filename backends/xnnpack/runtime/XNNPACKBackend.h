#pragma once

#include <executorch/runtime/platform/compiler.h>
#include <string>

namespace executorch::backends::xnnpack {
  const std::string enable_workspace_sharing_option_key = "enable_workspace_sharing";
}
