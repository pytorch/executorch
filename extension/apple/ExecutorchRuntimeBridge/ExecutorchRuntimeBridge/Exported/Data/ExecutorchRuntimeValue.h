// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifdef __cplusplus
 #import <executorch/extension/module/module.h>
 #import <executorch/runtime/core/evalue.h>
#endif

#import <ModelRunnerDataKit/ModelRunnerDataKit-Swift.h>

#import "ExecutorchRuntimeTensorValue.h"

NS_ASSUME_NONNULL_BEGIN

@interface ExecutorchRuntimeValue : NSObject <ModelRuntimeValueBridging>

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithTensor:(ExecutorchRuntimeTensorValue *)tensorValue;

#ifdef __cplusplus
- (instancetype)initWithEValue:(torch::executor::EValue)value NS_DESIGNATED_INITIALIZER;
- (torch::executor::EValue)getBackedValue;
#endif

@end

NS_ASSUME_NONNULL_END
