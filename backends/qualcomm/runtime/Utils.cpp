/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Utils.h>
namespace torch {
namespace executor {
namespace qnn {
const std::map<QcomChipset, HtpInfo>& PopulateSocInfoTable() {
  static const std::map<QcomChipset, HtpInfo> soc_info_map{
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
      {QcomChipset::SA8295,
       {QcomChipset::SA8295,
        QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V68,
        "SA8295",
        8}},
  };
  return soc_info_map;
}
} // namespace qnn
} // namespace executor
} // namespace torch
