//==============================================================================
//
// Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 * @file
 * @brief   QNN System Common API component
 *
 *          A header which contains common types shared by QNN system components.
 *          This simplifies the cross-inclusion of headers.
 */

#ifndef QNN_SYSTEM_COMMON_H
#define QNN_SYSTEM_COMMON_H

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

// libQnnSystem.so system interface provider name
#define QNN_SYSTEM_INTERFACE_PROVIDER_NAME "SYSTEM_QTI_AISW"

// Macro controlling visibility of QNN_SYSTEM API
#ifndef QNN_SYSTEM_API
#define QNN_SYSTEM_API
#endif

// Provide values to use for API version.
#define QNN_SYSTEM_API_VERSION_MAJOR 1
#define QNN_SYSTEM_API_VERSION_MINOR 1
#define QNN_SYSTEM_API_VERSION_PATCH 0

// Error code space assigned to system API components
#define QNN_SYSTEM_CONTEXT_MIN_ERROR QNN_MIN_ERROR_SYSTEM
#define QNN_SYSTEM_CONTEXT_MAX_ERROR (QNN_SYSTEM_CONTEXT_MIN_ERROR + 999)

//=============================================================================
// Data Types
//=============================================================================

//=============================================================================
// Public Functions
//=============================================================================

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_COMMON_H
