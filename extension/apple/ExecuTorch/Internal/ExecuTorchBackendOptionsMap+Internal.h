/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Internal extension header exposing the underlying C++ map to other ObjC++
// translation units in this module (e.g. ExecuTorchModule.mm). Not part of the
// public umbrella header. The C++ types in the method signatures mean this
// header is ObjC++-only — guarded against accidental import from a `.m` file.

#import "ExecuTorchBackendOptionsMap.h"

#ifdef __cplusplus

#import <executorch/runtime/backend/backend_options_map.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExecuTorchBackendOptionsMap (Internal)

/**
 * Pointer to the underlying C++ `LoadBackendOptionsMap`. The map is owned by
 * the receiver; callers must not retain or destroy it directly. Lifetime is
 * tied to the lifetime of this `ExecuTorchBackendOptionsMap` instance.
 */
- (const executorch::runtime::LoadBackendOptionsMap *)cppMap;

@end

NS_ASSUME_NONNULL_END

#endif  // __cplusplus
