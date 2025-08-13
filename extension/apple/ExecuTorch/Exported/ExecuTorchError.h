/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ExecuTorchErrorDomain NS_SWIFT_NAME(ErrorDomain);

/**
 * Enum to define the error codes.
 * Values can be a subset, but must numerically match exactly those defined in
 * runtime/core/error.h
 */
typedef NS_ERROR_ENUM(ExecuTorchErrorDomain, ExecuTorchErrorCode) {
  // System errors.
  ExecuTorchErrorCodeOk                              = 0,
  ExecuTorchErrorCodeInternal                        = 1,
  ExecuTorchErrorCodeInvalidState                    = 2,
  ExecuTorchErrorCodeEndOfMethod                     = 3,

  // Logical errors.
  ExecuTorchErrorCodeNotSupported                    = 16,
  ExecuTorchErrorCodeNotImplemented                  = 17,
  ExecuTorchErrorCodeInvalidArgument                 = 18,
  ExecuTorchErrorCodeInvalidType                     = 19,
  ExecuTorchErrorCodeOperatorMissing                 = 20,

  // Registration errors.
  ExecuTorchErrorCodeRegistrationExceedingMaxKernels = 21,
  ExecuTorchErrorCodeRegistrationAlreadyRegistered   = 22,

  // Resource errors.
  ExecuTorchErrorCodeNotFound                        = 32,
  ExecuTorchErrorCodeMemoryAllocationFailed          = 33,
  ExecuTorchErrorCodeAccessFailed                    = 34,
  ExecuTorchErrorCodeInvalidProgram                  = 35,
  ExecuTorchErrorCodeInvalidExternalData             = 36,
  ExecuTorchErrorCodeOutOfResources                  = 37,

  // Delegate errors.
  ExecuTorchErrorCodeDelegateInvalidCompatibility    = 48,
  ExecuTorchErrorCodeDelegateMemoryAllocationFailed  = 49,
  ExecuTorchErrorCodeDelegateInvalidHandle           = 50,
} NS_SWIFT_NAME(ErrorCode);

/**
 * Returns a brief error description for the given error code.
 *
 * @param code An ExecuTorchErrorCode value representing the error code.
 * @return An NSString containing the error description.
 */
FOUNDATION_EXPORT
__attribute__((deprecated("This API is experimental.")))
NSString *ExecuTorchErrorDescription(ExecuTorchErrorCode code)
    NS_SWIFT_NAME(ErrorDescription(_:));

/**
 * Create an NSError in the ExecuTorch domain for the given code.
 *
 * @param code The ExecuTorchErrorCode to wrap.
 * @return An NSError with ExecuTorchErrorDomain, the specified code, and a localized description.
 */
FOUNDATION_EXPORT
NS_RETURNS_RETAINED
__attribute__((deprecated("This API is experimental.")))
NSError *ExecuTorchErrorWithCode(ExecuTorchErrorCode code)
    NS_SWIFT_NAME(Error(code:));

/**
 * Create an NSError in the ExecuTorch domain for the given code.
 *
 * @param code The ExecuTorchErrorCode to wrap.
 * @param description Additional error description.
 * @return An NSError with ExecuTorchErrorDomain, the specified code, and a localized description.
 */
 FOUNDATION_EXPORT
 NS_RETURNS_RETAINED
 __attribute__((deprecated("This API is experimental.")))
 NSError *ExecuTorchErrorWithCodeAndDescription(ExecuTorchErrorCode code, NSString * __nullable description)
     NS_SWIFT_NAME(Error(code:description:));

NS_ASSUME_NONNULL_END
