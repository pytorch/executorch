/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchDumpTracer.h"

#import "ExecuTorchDumpTracer+Internal.h"

#import <ExecuTorch/ExecuTorchEventTracer+Internal.h>

using executorch::etdump::ETDumpGen;

@implementation ExecuTorchDumpTracer {
  // A typed pointer to the generator the base class owns through its C++ tracer,
  // so the recorder can read the trace back without a downcast. It stays valid
  // for as long as the module the tracer was given to lives.
  ETDumpGen *_generator;
}

// init chains to the base class's real designated initializer, initWithCppTracer:,
// which takes a C++ type and so lives in an Objective-C++ category rather than the
// public interface. Clang cannot see a category initializer as designated, so it
// wrongly flags this chain; the pattern is deliberate and correct.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-designated-initializers"

- (instancetype)init {
  auto generator = std::make_unique<ETDumpGen>();
  _generator = generator.get();
  return [super initWithCppTracer:std::move(generator)];
}

#pragma clang diagnostic pop

- (ETDumpGen *)generator {
  return _generator;
}

@end
