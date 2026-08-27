/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchModule.h"

#ifdef __cplusplus
#import <memory>

#import <executorch/runtime/core/event_tracer.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExecuTorchModule (Internal)

/**
 * Creates a module that records events through the given tracer.
 *
 * A class method rather than an initializer, because a category cannot add a
 * designated initializer and the compiler would then treat the public ones as
 * convenience initializers that fail to chain.
 *
 * @param filePath The path to the model file.
 * @param dataFilePaths The paths to the model data files.
 * @param loadMode The load mode to use.
 * @param eventTracer The tracer the module records through, whose ownership it
 * takes.
 * @return A new module that owns the tracer.
 */
+ (instancetype)moduleWithFilePath:(NSString *)filePath
                     dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                          loadMode:(ExecuTorchModuleLoadMode)loadMode
                       eventTracer:
                           (std::unique_ptr<executorch::runtime::EventTracer>)
                               eventTracer;

/**
 * The tracer the module records through, or null if it has none.
 *
 * The module owns the tracer, so this pointer is valid only while the module is.
 */
- (nullable executorch::runtime::EventTracer *)eventTracer;

@end

NS_ASSUME_NONNULL_END

#endif // __cplusplus
