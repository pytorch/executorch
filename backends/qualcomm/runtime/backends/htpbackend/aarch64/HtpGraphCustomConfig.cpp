/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraphCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {
std::vector<QnnGraph_CustomConfig_t>
HtpGraphCustomConfig::CreateGraphCustomConfig(
    const SocInfo* qcom_target_soc_info) {
  return CreateGraphCustomConfigCommon(qcom_target_soc_info, 1);
}
} // namespace qnn
} // namespace backends
} // namespace executorch
