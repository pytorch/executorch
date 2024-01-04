//==============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QNN_HTP_PROPERTY_H
#define QNN_HTP_PROPERTY_H

#include "QnnProperty.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
/**
 * @brief Property key for determining whether a backend supports unsigned pd.
 */
#define QNN_PROPERTY_CUSTOM_HTP_UNSIGNED_PD_SUPPORT QNN_PROPERTY_GROUP_CUSTOM + 1

#ifdef __cplusplus
}
#endif

#endif  // QNN_HTP_PROPERTY_H
