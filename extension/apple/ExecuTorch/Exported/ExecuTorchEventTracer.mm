/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchEventTracer.h"

#import <executorch/runtime/core/event_tracer.h>

using executorch::runtime::EventTracer;

@implementation ExecuTorchEventTracer {
  std::unique_ptr<EventTracer> _tracer;
}

- (instancetype)initWithNativeInstance:(void *)nativeInstance {
  ET_CHECK(nativeInstance);
  if (self = [super init]) {
    _tracer = std::move(
        *reinterpret_cast<std::unique_ptr<EventTracer> *>(nativeInstance));
    ET_CHECK(_tracer);
  }
  return self;
}

- (void *)nativeInstance {
  return &_tracer;
}

@end
