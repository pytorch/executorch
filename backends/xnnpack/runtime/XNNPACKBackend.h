#pragma once

#include <executorch/runtime/platform/compiler.h>

namespace executorch::backends::xnnpack {
ET_EXPERIMENTAL void set_workspace_sharing_enabled(bool enable);
}
