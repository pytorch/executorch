/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchError.h"

NSErrorDomain const ExecuTorchErrorDomain = @"org.pytorch.executorch.error";

NSString *ExecuTorchErrorDescription(ExecuTorchErrorCode code) {
  switch (code) {
    case ExecuTorchErrorCodeOk:
      return @"";
    case ExecuTorchErrorCodeInternal:
      return @"Internal error";
    case ExecuTorchErrorCodeInvalidState:
      return @"Invalid executor state";
    case ExecuTorchErrorCodeEndOfMethod:
      return @"No more execution steps";
    case ExecuTorchErrorCodeNotSupported:
      return @"Operation not supported";
    case ExecuTorchErrorCodeNotImplemented:
      return @"Operation not implemented";
    case ExecuTorchErrorCodeInvalidArgument:
      return @"Invalid argument";
    case ExecuTorchErrorCodeInvalidType:
      return @"Invalid type";
    case ExecuTorchErrorCodeOperatorMissing:
      return @"Operator missing";
    case ExecuTorchErrorCodeRegistrationExceedingMaxKernels:
      return @"Exceeded maximum number of kernels";
    case ExecuTorchErrorCodeRegistrationAlreadyRegistered:
      return @"Kernel is already registered";
    case ExecuTorchErrorCodeNotFound:
      return @"Resource not found";
    case ExecuTorchErrorCodeMemoryAllocationFailed:
      return @"Memory allocation failed";
    case ExecuTorchErrorCodeAccessFailed:
      return @"Access failed";
    case ExecuTorchErrorCodeInvalidProgram:
      return @"Invalid program contents";
    case ExecuTorchErrorCodeInvalidExternalData:
      return @"Invalid external data";
    case ExecuTorchErrorCodeOutOfResources:
      return @"Out of resources";
    case ExecuTorchErrorCodeDelegateInvalidCompatibility:
      return @"Delegate version incompatible";
    case ExecuTorchErrorCodeDelegateMemoryAllocationFailed:
      return @"Delegate memory allocation failed";
    case ExecuTorchErrorCodeDelegateInvalidHandle:
      return @"Delegate handle invalid";
    default:
      return @"Unknown error";
  }
}

NSError *ExecuTorchErrorWithCode(ExecuTorchErrorCode code) {
  return ExecuTorchErrorWithCodeAndDescription(code, nil);
}

NSError *ExecuTorchErrorWithCodeAndDescription(ExecuTorchErrorCode code, NSString * __nullable description) {
  return [[NSError alloc] initWithDomain:ExecuTorchErrorDomain
                                    code:code
                                userInfo:@{
    NSLocalizedDescriptionKey:
      description.length > 0
        ? [ExecuTorchErrorDescription(code) stringByAppendingFormat:@": %@", description]
        : ExecuTorchErrorDescription(code)
  }];
}
