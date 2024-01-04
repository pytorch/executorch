//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for HTP backend
 */

#ifndef QNN_HTP_COMMON_H
#define QNN_HTP_COMMON_H

#include "QnnCommon.h"

/// HTP Backend identifier
#define QNN_BACKEND_ID_HTP 6

/// HTP interface provider
#define QNN_HTP_INTERFACE_PROVIDER_NAME "HTP_QTI_AISW"

// HTP API Version values
#define QNN_HTP_API_VERSION_MAJOR 5
#define QNN_HTP_API_VERSION_MINOR 16
#define QNN_HTP_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for HTP backend
#define QNN_HTP_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
        QNN_API_VERSION_MAJOR,        /*coreApiVersion.major*/   \
        QNN_API_VERSION_MINOR,        /*coreApiVersion.major*/   \
        QNN_API_VERSION_PATCH         /*coreApiVersion.major*/   \
    },                                                           \
    {                                                            \
      QNN_HTP_API_VERSION_MAJOR,     /*backendApiVersion.major*/ \
      QNN_HTP_API_VERSION_MINOR,     /*backendApiVersion.minor*/ \
      QNN_HTP_API_VERSION_PATCH      /*backendApiVersion.patch*/ \
    }                                                            \
  }

// clang-format on

// DSP Context blob Version values
#define QNN_HTP_CONTEXT_BLOB_VERSION_MAJOR 3
#define QNN_HTP_CONTEXT_BLOB_VERSION_MINOR 1
#define QNN_HTP_CONTEXT_BLOB_VERSION_PATCH 0

/* ==== CDSP Security Library Versioning ==== */
/* ==== This information is only intended for OEMs ==== */

/* Security versioning for DSP libraries is supported V73 onwards */
#define QNN_HTP_NATIVE_LIB_SECURITY_VERSIONING_MIN_ARCH 73

/* Here we will define CDSP library versions for different targets
 * Version is increased whenever there is a security fix from CDSP
 * The versioning will start from 1.0.0 for each new target
 * */

/* V73 Security Issues:
 * List of security issues fixed for V73 and the fixed version
 * */
#define QNN_HTP_V73_NATIVE_LIB_SECURITY_VERSION_MAJOR 1
#define QNN_HTP_V73_NATIVE_LIB_SECURITY_VERSION_MINOR 0
#define QNN_HTP_V73_NATIVE_LIB_SECURITY_VERSION_PATCH 0

/* V75 Security Issues:
 * List of security issues fixed for V75 and the fixed version
 * */
// HTP Native library version values for V75
#define QNN_HTP_V75_NATIVE_LIB_SECURITY_VERSION_MAJOR 1
#define QNN_HTP_V75_NATIVE_LIB_SECURITY_VERSION_MINOR 0
#define QNN_HTP_V75_NATIVE_LIB_SECURITY_VERSION_PATCH 0


#endif  // QNN_HTP_COMMON_H
