/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "QnnCommon.h"

// QnnCommon.h exposes the QNN API ABI version, not the QAIRT SDK product
// release. Use this helper for compile-time header/API availability checks.
#define QNN_EXECUTORCH_QNN_API_VERSION_AT_LEAST(major, minor) \
  ((QNN_API_VERSION_MAJOR > (major)) ||                       \
   (QNN_API_VERSION_MAJOR == (major) && QNN_API_VERSION_MINOR >= (minor)))

// Product-release requirement: FCB requires QAIRT SDK 2.48 or newer. That
// release provides QNN API 2.37, which adds the required core, System DLC,
// and HTP declarations. Keep this mapping here when a new SDK release changes
// the FCB interface requirement.
#define QNN_EXECUTORCH_FCB_MIN_QAIRT_SDK_VERSION "2.48"

#if QNN_EXECUTORCH_QNN_API_VERSION_AT_LEAST(2, 37)
#define QNN_EXECUTORCH_SUPPORTS_FCB 1
#else
#define QNN_EXECUTORCH_SUPPORTS_FCB 0
#endif
