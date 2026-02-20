/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpDevicePlatformInfoConfig.h>
namespace executorch {
namespace backends {
namespace qnn {
std::vector<QnnDevice_PlatformInfo_t*>
HtpDevicePlatformInfoConfig::CreateDevicePlatformInfo(
    const SocInfo* /*qcom_target_soc_info*/) {
  return {};
}
} // namespace qnn
} // namespace backends
} // namespace executorch
