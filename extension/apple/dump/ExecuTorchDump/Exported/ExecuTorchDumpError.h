/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ExecuTorchDumpErrorDomain NS_SWIFT_NAME(DumpErrorDomain);

/**
 * Errors reported while recording a profile.
 */
typedef NS_ERROR_ENUM(ExecuTorchDumpErrorDomain, ExecuTorchDumpErrorCode){
    /**
     * The linked runtime was built without tracing, so nothing can be recorded.
     * Check `ExecuTorchDump.available` before creating one.
     */
    ExecuTorchDumpErrorCodeUnavailable = 1,
    /**
     * Nothing has been recorded since the last read. Run a method first.
     */
    ExecuTorchDumpErrorCodeNoData = 2,
} NS_SWIFT_NAME(DumpError);

NS_ASSUME_NONNULL_END
