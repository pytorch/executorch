/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpBackendCache.h>
#include "HTP/QnnHtpSystemContext.h"

namespace torch {
namespace executor {
namespace qnn {
Error HtpBackendCache::RetrieveBackendBinaryInfo(
    const QnnSystemContext_BinaryInfo_t* binaryinfo) {
  QnnHtpSystemContext_HwBlobInfo_t* htp_hwblobinfo = nullptr;

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV1.hwInfoBlob);
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV2.hwInfoBlob);
  } else {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN BinaryInfo version %d.", binaryinfo->version);
    return Error::Internal;
  }

  if (htp_hwblobinfo == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Htp hardware blob information is not found in binary information.");
    return Error::Ok;
  }

  if (htp_hwblobinfo->version ==
      QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1) {
    spill_fill_buf_ =
        (*htp_hwblobinfo).contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
  } else {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN Htp hw blob info version %d.", htp_hwblobinfo->version);
    return Error::Internal;
  }

  return Error::Ok;
}

} // namespace qnn
} // namespace executor
} // namespace torch
