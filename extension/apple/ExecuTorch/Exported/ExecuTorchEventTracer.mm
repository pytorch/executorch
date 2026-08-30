/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchEventTracer.h"

#import <executorch/extension/apple/ExecuTorch/Internal/ExecuTorchEventTracer+Internal.h>

using executorch::runtime::EventTracer;

@implementation ExecuTorchEventTracer {
  std::unique_ptr<EventTracer> _tracer;
}

- (instancetype)initWithCppTracer:(std::unique_ptr<EventTracer>)tracer {
  self = [super init];
  if (self) {
    _tracer = std::move(tracer);
  }
  return self;
}

- (std::unique_ptr<EventTracer>)takeCppTracer {
  return std::move(_tracer);
}

@end
