//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN Saver Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for Saver backend
 */

#ifndef QNN_SAVER_COMMON_H
#define QNN_SAVER_COMMON_H

#include "QnnCommon.h"

/// Saver Backend identifier
#define QNN_BACKEND_ID_SAVER 2

/// Saver interface provider
#define QNN_SAVER_INTERFACE_PROVIDER_NAME "SAVER_QTI_AISW"

// Saver API Version values
#define QNN_SAVER_API_VERSION_MAJOR 1
#define QNN_SAVER_API_VERSION_MINOR 0
#define QNN_SAVER_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for Saver backend
#define QNN_SAVER_API_VERSION_INIT                               \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_SAVER_API_VERSION_MAJOR, /*backendApiVersion.major*/   \
      QNN_SAVER_API_VERSION_MINOR, /*backendApiVersion.minor*/   \
      QNN_SAVER_API_VERSION_PATCH  /*backendApiVersion.patch*/   \
    }                                                            \
  }

// clang-format on

#endif  // QNN_SAVER_COMMON_H