/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

#import <ExecuTorch/ExecuTorchModule.h>

#import "ExecuTorchETDumpError.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Records what a model does while it runs: which methods and operators
 * executed, and how long each took.
 *
 * Load a model through this instead of through `Module` directly, run methods
 * on the `module` it owns, then take the recorded trace. The trace is an
 * ETDump, which the ExecuTorch developer tools read.
 *
 * Events accumulate across runs, and taking the trace completes it, so take it
 * once per report rather than after every call.
 */
NS_SWIFT_NAME(ETDump)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchETDump : NSObject

/**
 * Whether the linked runtime records events at all.
 *
 * Tracing is a build-time choice: the frameworks published for Apple platforms
 * enable it, a runtime built from source does so only when asked. When this is
 * NO, creating an instance fails rather than returning one that records
 * nothing.
 */
@property(class, readonly, getter=isAvailable)
    BOOL available NS_SWIFT_NAME(isAvailable);

/**
 * The module being profiled. Run methods on this.
 *
 * Owned by the recorder, because the two share the tracer that connects them.
 */
@property(readonly) ExecuTorchModule *module;

/**
 * Creates a recorder for a model, along with the module that reports to it.
 *
 * @param filePath The path to the .pte file.
 * @param dataFilePaths Paths to external tensor data files, if the model has
 * any.
 * @param loadMode How the file is read.
 * @param error On failure, describes why, most often that the runtime was built
 * without tracing.
 * @return A recorder, or nil.
 */
- (nullable instancetype)initWithFilePath:(NSString *)filePath
                            dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                                 loadMode:(ExecuTorchModuleLoadMode)loadMode
                                    error:(NSError **)error
    NS_SWIFT_NAME(init(filePath:dataFilePaths:loadMode:))
        NS_DESIGNATED_INITIALIZER;

/**
 * Creates a recorder for a model, reading the file in the default way.
 *
 * @param filePath The path to the .pte file.
 * @param error On failure, describes why.
 * @return A recorder, or nil.
 */
- (nullable instancetype)initWithFilePath:(NSString *)filePath
                                    error:(NSError **)error
    NS_SWIFT_NAME(init(filePath:));

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/**
 * Takes the trace recorded so far, in the ETDump format.
 *
 * Completing the trace ends it, so this returns each recorded span once and a
 * subsequent call reports that there is nothing new. Run a method first.
 *
 * @param error On failure, describes why, most often that nothing has run yet.
 * @return The trace, or nil.
 */
- (nullable NSData *)takeDataWithError:(NSError **)error
    NS_SWIFT_NAME(takeData());

/**
 * Takes the trace recorded so far and writes it to a file.
 *
 * @param path The file to write. An existing file is replaced.
 * @param error On failure, describes why.
 * @return YES if a trace was written.
 */
- (BOOL)takeDataToFile:(NSString *)path
                 error:(NSError **)error NS_SWIFT_NAME(takeData(toFile:));

@end

NS_ASSUME_NONNULL_END
