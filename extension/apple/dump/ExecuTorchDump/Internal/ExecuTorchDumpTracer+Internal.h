/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchDumpTracer.h"

#ifdef __cplusplus

#import <executorch/devtools/etdump/etdump_flatcc.h>

NS_ASSUME_NONNULL_BEGIN

// Internal extension exposing the underlying ETDump generator to other ObjC++
// translation units in this framework (ExecuTorchDump.mm). Not part of the
// public umbrella header. The C++ type in the signature makes this ObjC++-only.
@interface ExecuTorchDumpTracer (Internal)

// The generator this tracer records through. Owned by the base class's C++
// tracer, so it stays valid for as long as the module the tracer was given to.
- (executorch::etdump::ETDumpGen *)generator;

@end

NS_ASSUME_NONNULL_END

#endif // __cplusplus
