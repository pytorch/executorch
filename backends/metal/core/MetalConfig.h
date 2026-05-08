/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalConfig.h — single source of truth for compile-time feature gates.
//
// Two macros are defined here and only here:
//
//   ET_METAL4_AVAILABLE  — SDK-level availability gate. 1 iff the
//                          compile target is macOS ≥ 15 OR iOS ≥ 18 (the
//                          first SDKs that expose MTLResidencySet, MTL4
//                          types, etc.). This is independent of whether
//                          the build OPTS into the MTL4 dispatch path.
//
//   ET_METAL4_ENABLE     — Build-flag gate set by the CMake option
//                          METAL_V2_USE_MTL4. When 1, the MTL4 dispatch
//                          path is compiled in alongside MTL3. When 0,
//                          only the MTL3 path is built.
//
// All headers and TUs in core/ MUST include this header instead of
// re-defining the macros locally — duplicate definitions historically
// drifted (e.g., one that checked only macOS, leaving iOS broken).
//===----------------------------------------------------------------------===//

#include <Availability.h>

// SDK availability: macOS 15+ or iOS 18+. Both branches are required because
// ResidencyManager and similar classes are reachable from iOS-only TUs.
#if !defined(ET_METAL4_AVAILABLE)
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000)
#define ET_METAL4_AVAILABLE 1
#else
#define ET_METAL4_AVAILABLE 0
#endif
#endif

// Build-flag opt-in for the MTL4 dispatch path. CMake passes this
// PUBLIC for every consumer of metal_v2 (see CMakeLists.txt). Default 0
// when the build system hasn't set it (e.g., header-only consumers).
#if !defined(ET_METAL4_ENABLE)
#define ET_METAL4_ENABLE 0
#endif
