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

- (instancetype)init {
  auto generator = std::make_unique<ETDumpGen>();
  _generator = generator.get();
  return [super initWithCppTracer:std::move(generator)];
}

- (ETDumpGen *)generator {
  return _generator;
}

@end
