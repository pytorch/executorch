/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if __has_include(<ExecuTorch/ExecuTorchEventTracer.h>)
#import <ExecuTorch/ExecuTorchEventTracer.h>
#else
#import "ExecuTorchEventTracer.h"
#endif

#ifdef __cplusplus

#import <memory>

#import <executorch/runtime/core/event_tracer.h>

NS_ASSUME_NONNULL_BEGIN

// Internal extension header exposing the underlying C++ tracer to other ObjC++
// translation units in this module (ExecuTorchModule.mm) and to concrete tracer
// subclasses in other frameworks. Not part of the public umbrella header. The
// C++ types in the method signatures mean this header is ObjC++-only, guarded
// against accidental import from a `.m` file.
@interface ExecuTorchEventTracer (Internal)

/**
 * Wraps a C++ tracer. A concrete subclass calls this to hand its implementation
 * to the base class, which owns it until a module takes it.
 */
- (instancetype)initWithCppTracer:
    (std::unique_ptr<executorch::runtime::EventTracer>)tracer;

/**
 * Moves the C++ tracer out of the receiver. Called once, by the module that
 * takes ownership of the tracer at construction. The receiver holds nothing
 * after this.
 */
- (std::unique_ptr<executorch::runtime::EventTracer>)takeCppTracer;

@end

NS_ASSUME_NONNULL_END

#endif // __cplusplus
