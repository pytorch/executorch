/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpBackendCache.h>
#include "HTP/QnnHtpSystemContext.h"

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

Error HtpBackendCache::RetrieveBackendBinaryInfo(
    const QnnSystemContext_BinaryInfo_t* binaryinfo) {
  QnnHtpSystemContext_HwBlobInfo_t* htp_hwblobinfo = nullptr;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  QnnHtpSystemContext_GraphBlobInfo_t* htp_graphblobinfo = nullptr;
#endif

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV1.hwInfoBlob);
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV2.hwInfoBlob);
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    htp_graphblobinfo = static_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV3.graphs->graphInfoV3.graphBlobInfo);
#endif
  } else {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN BinaryInfo version %d.", binaryinfo->version);
    return Error::Internal;
  }

  if (htp_hwblobinfo) {
    if (htp_hwblobinfo->version ==
        QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1) {
      spill_fill_buf_ =
          (*htp_hwblobinfo).contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
    } else {
      QNN_EXECUTORCH_LOG_WARN(
          "Unknown QNN Htp hw blob info version %d.", htp_hwblobinfo->version);
      return Error::Internal;
    }
  }

#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  if (htp_graphblobinfo) {
    if (htp_graphblobinfo->version ==
        QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
      spill_fill_buf_ =
          (*htp_graphblobinfo).contextBinaryGraphBlobInfoV1.spillFillBufferSize;
    } else {
      QNN_EXECUTORCH_LOG_WARN(
          "Unknown QNN Htp graph blob info version %d.",
          htp_graphblobinfo->version);
      return Error::Internal;
    }
  }
#endif

  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
