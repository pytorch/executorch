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
