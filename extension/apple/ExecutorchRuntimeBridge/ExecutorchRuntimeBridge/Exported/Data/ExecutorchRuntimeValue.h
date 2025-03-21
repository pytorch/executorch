/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __cplusplus
 #import <executorch/extension/module/module.h>
 #import <executorch/runtime/core/evalue.h>
#endif

#import <RuntimeBridgingCore/RuntimeBridgingCore-Swift.h>

#import "ExecutorchRuntimeTensorValue.h"

NS_ASSUME_NONNULL_BEGIN

@interface ExecutorchRuntimeValue : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithTensor:(ExecutorchRuntimeTensorValue *)tensorValue;

#ifdef __cplusplus
- (instancetype)initWithEValue:(torch::executor::EValue)value NS_DESIGNATED_INITIALIZER;
- (torch::executor::EValue)getBackedValue;
#endif

#pragma mark -
- (ExecutorchRuntimeTensorValue *_Nullable)asTensorValueAndReturnError:(NSError * _Nullable * _Nullable)error SWIFT_WARN_UNUSED_RESULT;

@end

NS_ASSUME_NONNULL_END
