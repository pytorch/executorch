/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_error.h"

#import "ETCoreAIBridge.h"

#include <executorch/runtime/platform/log.h>

namespace executorch::backends::coreai {

runtime::Error execution_error(NSError* error) {
  ET_LOG(
      Error,
      "Core AI execution error (%s, %ld): %s",
      error.domain.UTF8String,
      static_cast<long>(error.code),
      error.description.UTF8String);
  if ([error.domain isEqualToString:ETCoreAIErrorDomain]) {
    switch (error.code) {
      case ETCoreAIErrorInvalidModel:
        return runtime::Error::InvalidProgram;
      case ETCoreAIErrorUnsupported:
        return runtime::Error::NotSupported;
      case ETCoreAIErrorInvalidArgument:
        return runtime::Error::InvalidArgument;
      default:
        break;
    }
  }
  return runtime::Error::Internal;
}

runtime::Error acquisition_error(NSError* error) {
  ET_LOG(
      Error,
      "Core AI acquisition error (%s, %ld): %s",
      error.domain.UTF8String,
      static_cast<long>(error.code),
      error.description.UTF8String);
  return runtime::Error::Internal;
}

} // namespace executorch::backends::coreai
