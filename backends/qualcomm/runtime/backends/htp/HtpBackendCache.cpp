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
  std::vector<QnnHtpSystemContext_GraphBlobInfo_t*> htp_graphblobinfos;
  std::uint32_t num_graphs;

#endif

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV1.hwInfoBlob);
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    htp_hwblobinfo = static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(
        binaryinfo->contextBinaryInfoV2.hwInfoBlob);
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21)
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    num_graphs = binaryinfo->contextBinaryInfoV3.numGraphs;
    for (size_t i = 0; i < num_graphs; ++i) {
      htp_graphblobinfos.push_back(
          static_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(
              binaryinfo->contextBinaryInfoV3.graphs[i]
                  .graphInfoV3.graphBlobInfo));
    }
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
  if (htp_graphblobinfos.size() > 0) {
    // After version 2.21, we need to get spill fill buffer size from graph
    // blob info instead of hw blob info. If there are multiple graphs, we
    // should use the max value among all graphs.
    if (htp_graphblobinfos[0]->version ==
        QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
      for (size_t i = 0; i < num_graphs; ++i) {
        uint64_t spill_fill_buf =
            (*htp_graphblobinfos[i])
                .contextBinaryGraphBlobInfoV1.spillFillBufferSize;
        if (spill_fill_buf > spill_fill_buf_) {
          spill_fill_buf_ = spill_fill_buf;
        }
      }
    } else {
      QNN_EXECUTORCH_LOG_WARN(
          "Unknown QNN Htp graph blob info version %d.",
          htp_graphblobinfos[0]->version);
      return Error::Internal;
    }
  }
#endif

  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
