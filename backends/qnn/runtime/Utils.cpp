/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qnn/runtime/Utils.h>
namespace torch {
namespace executor {
namespace qnn {
const std::map<QcomModel, HtpInfo>& PopulateSocInfoTable() {
  static const std::map<QcomModel, HtpInfo> soc_info_map{
      {QcomModel::SM8550,
       {QcomModel::SM8550, QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V73,
        "SM8550", 8}},
      {QcomModel::SM8450,
       {QcomModel::SM8450, QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V69,
        "SM8450", 8}},
      {QcomModel::SM8475,
       {QcomModel::SM8475, QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V69,
        "SM8475", 8}},
  };
  return soc_info_map;
}
}  // namespace qnn
}  // namespace executor
}  // namespace torch
