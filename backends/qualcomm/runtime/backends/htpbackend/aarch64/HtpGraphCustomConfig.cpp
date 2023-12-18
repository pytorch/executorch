/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraphCustomConfig.h>
namespace torch {
namespace executor {
namespace qnn {
std::vector<QnnGraph_CustomConfig_t>
HtpGraphCustomConfig::CreateGraphCustomConfig(
    const HtpInfo& /*qcom_target_soc_info*/) {
  return {};
}
} // namespace qnn
} // namespace executor
} // namespace torch
