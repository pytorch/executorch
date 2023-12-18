/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Utils.h>

#include <executorch/backends/qualcomm/runtime/Logging.h>
namespace torch {
namespace executor {
namespace qnn {
const std::unordered_map<QcomChipset, HtpInfo>& PopulateSocInfoTable() {
  static const std::unordered_map<QcomChipset, HtpInfo> soc_info_map{
      {QcomChipset::SM8550,
       {QcomChipset::SM8550,
        QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V73,
        "SM8550",
        8}},
      {QcomChipset::SM8450,
       {QcomChipset::SM8450,
        QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V69,
        "SM8450",
        8}},
      {QcomChipset::SM8475,
       {QcomChipset::SM8475,
        QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V69,
        "SM8475",
        8}},
  };
  return soc_info_map;
}

HtpInfo GetHtpInfo(const QcomChipset& soc) {
  const std::unordered_map<QcomChipset, HtpInfo>& soc_to_info =
      PopulateSocInfoTable();
  auto soc_info_pair = soc_to_info.find(soc);

  if (soc_info_pair == soc_to_info.end()) {
    QcomChipset default_soc_model = QcomChipset::SM8550;
    QNN_EXECUTORCH_LOG(
        kLogLevelWarn,
        "[Qnn ExecuTorch] Failed to get soc info for "
        "soc model %d. Using default soc_model=%d",
        soc,
        default_soc_model);
    soc_info_pair = soc_to_info.find(default_soc_model);
  }

  return soc_info_pair->second;
}

} // namespace qnn
} // namespace executor
} // namespace torch
