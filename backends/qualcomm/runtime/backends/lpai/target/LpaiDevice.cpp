/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiDevice.h>

namespace executorch {
namespace backends {
namespace qnn {

Error LpaiDevice::Configure() {
#ifndef __hexagon__
  return QnnDevice::Configure();
#else
  return Error::Ok;
#endif
}

} // namespace qnn
} // namespace backends
} // namespace executorch
