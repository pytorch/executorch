/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDeviceCustomConfig.h>
namespace torch {
namespace executor {
namespace qnn {
std::vector<QnnDevice_CustomConfig_t>
HtpDeviceCustomConfig::CreateDeviceCustomConfig(
    const SocInfo* /*qcom_target_soc_info*/) {
  return {};
}
} // namespace qnn
} // namespace executor
} // namespace torch
