/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ExecuTorchETDumpErrorDomain
    NS_SWIFT_NAME(ETDumpErrorDomain);

/**
 * Errors reported while recording a profile.
 */
typedef NS_ERROR_ENUM(ExecuTorchETDumpErrorDomain, ExecuTorchETDumpErrorCode){
    /**
     * The linked runtime was built without tracing, so nothing can be recorded.
     * Check `ExecuTorchETDump.available` before creating one.
     */
    ExecuTorchETDumpErrorCodeUnavailable = 1,
    /**
     * Nothing has been recorded since the last read. Run a method first.
     */
    ExecuTorchETDumpErrorCodeNoData = 2,
} NS_SWIFT_NAME(ETDumpError);

NS_ASSUME_NONNULL_END
