/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchValue.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Enum to define loading behavior.
 * Values can be a subset, but must numerically match exactly those defined in
 * extension/module/module.h
 */
typedef NS_ENUM(NSInteger, ExecuTorchModuleLoadMode) {
  ExecuTorchModuleLoadModeFile = 0,
  ExecuTorchModuleLoadModeMmap,
  ExecuTorchModuleLoadModeMmapUseMlock,
  ExecuTorchModuleLoadModeMmapUseMlockIgnoreErrors,
} NS_SWIFT_NAME(ModuleLoadMode);

NS_SWIFT_NAME(Module)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchModule : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
