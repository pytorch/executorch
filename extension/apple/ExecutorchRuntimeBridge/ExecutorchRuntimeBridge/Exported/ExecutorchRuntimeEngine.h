// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#import <Foundation/Foundation.h>

#import "ExecutorchRuntimeValue.h"

NS_ASSUME_NONNULL_BEGIN

@interface ExecutorchRuntimeEngine : NSObject

- (nonnull instancetype)init NS_UNAVAILABLE;
+ (nonnull instancetype)new NS_UNAVAILABLE;

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                           modelMethodName:(NSString *)modelMethodName
                                     error:(NSError * _Nullable * _Nullable)error NS_DESIGNATED_INITIALIZER;

- (nullable NSArray<ExecutorchRuntimeValue *> *)infer:(NSArray<ExecutorchRuntimeValue *> *)input
                                                error:(NSError * _Nullable * _Nullable)error NS_SWIFT_NAME(infer(input:));

@end

NS_ASSUME_NONNULL_END
