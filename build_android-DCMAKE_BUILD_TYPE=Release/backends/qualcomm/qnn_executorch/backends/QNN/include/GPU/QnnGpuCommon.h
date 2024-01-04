//==============================================================================
//
// Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines common QNN GPU macros.
 */

#ifndef QNN_GPU_COMMON_H
#define QNN_GPU_COMMON_H

#include "QnnCommon.h"

/// GPU Backend identifier
#define QNN_BACKEND_ID_GPU 4

/// GPU interface provider
#define QNN_GPU_INTERFACE_PROVIDER_NAME "GPU_QTI_AISW"

// GPU API Version values
#define QNN_GPU_API_VERSION_MAJOR 3
#define QNN_GPU_API_VERSION_MINOR 3
#define QNN_GPU_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for GPU backend
#define QNN_GPU_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_GPU_API_VERSION_MAJOR, /*backendApiVersion.major*/     \
      QNN_GPU_API_VERSION_MINOR, /*backendApiVersion.minor*/     \
      QNN_GPU_API_VERSION_PATCH  /*backendApiVersion.patch*/     \
    }                                                            \
  }

// clang-format on

#endif  // QNN_GPU_COMMON_H
