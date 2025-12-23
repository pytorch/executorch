/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnContext_CustomConfig_t>
GpuContextCustomConfig::CreateContextCustomConfig() {
  return {};
}

} // namespace qnn
} // namespace backends
} // namespace executorch
