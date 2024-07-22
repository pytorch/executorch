/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Suppress an unused variable. Copied from C10_UNUSED
#if defined(_MSC_VER) && !defined(__clang__)
#define VK_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define VK_UNUSED __attribute__((__unused__))
#endif //_MSC_VER
