/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraph.h>

namespace executorch {
namespace backends {
namespace qnn {

Error LpaiGraph::AfterConfigure(const std::string& graph_name) {
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
