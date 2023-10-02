/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>

#include <map>
#include <string>

#include "HTP/QnnHtpDevice.h"
namespace torch {
namespace executor {
namespace qnn {
class HtpInfo {
 public:
  HtpInfo()
      : HtpInfo(
            QcomChipset::UNKNOWN_SM,
            QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_NONE,
            "",
            0){};
  HtpInfo(
      QcomChipset socModel,
      QnnHtpDevice_Arch_t htpArch,
      std::string socName,
      size_t vtcmSizeinMB)
      : m_socModel(socModel),
        m_htpArch(htpArch),
        m_socName(std::move(socName)),
        m_vtcmSizeinMB(vtcmSizeinMB) {}
  QcomChipset m_socModel;
  QnnHtpDevice_Arch_t m_htpArch;
  std::string m_socName;
  size_t m_vtcmSizeinMB;
};

const std::map<QcomChipset, HtpInfo>& PopulateSocInfoTable();
} // namespace qnn
} // namespace executor
} // namespace torch
