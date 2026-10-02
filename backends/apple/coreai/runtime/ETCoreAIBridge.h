/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, ETCoreAIScalarType) {
  ETCoreAIScalarTypeFloat16 = 0,
  ETCoreAIScalarTypeFloat32 = 1,
};

typedef NS_ENUM(NSInteger, ETCoreAIErrorCode) {
  ETCoreAIErrorInvalidModel = 1,
  ETCoreAIErrorUnsupported = 2,
  ETCoreAIErrorInvalidArgument = 3,
  ETCoreAIErrorRuntime = 4,
};

FOUNDATION_EXPORT NSErrorDomain const ETCoreAIErrorDomain;

// Input storage is borrowed until session completion, never retained or freed.
@interface ETCoreAIInputTensor : NSObject
@property(nonatomic, readonly) const void *_Nullable bytes;
@property(nonatomic, readonly) NSUInteger byteCount;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) ETCoreAIScalarType scalarType;
- (instancetype)initWithBytes:(const void *_Nullable)bytes
                    byteCount:(NSUInteger)byteCount
                        shape:(NSArray<NSNumber *> *)shape
                   scalarType:(ETCoreAIScalarType)scalarType;
@end

// Outputs own densely packed bytes, independent of the executor's storage.
@interface ETCoreAITensor : NSObject
@property(nonatomic, readonly, copy) NSData *data;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) ETCoreAIScalarType scalarType;
- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                  scalarType:(ETCoreAIScalarType)scalarType;
@end

// The caller keeps inputs valid and unmodified until completion. All session
// input access must finish before completion is invoked, including on failure.
@protocol ETCoreAISession <NSObject>
- (void)executeInputs:(NSArray<ETCoreAIInputTensor *> *)inputs
           completion:(void (^)(
                          NSArray<ETCoreAITensor *> *_Nullable outputs,
                          NSError *_Nullable error))completion;
@end

// A prepared model pins the SDK entry. Sessions retain their own pin after binding.
@protocol ETCoreAIPreparedModel <NSObject>
- (NSData *)copyBookmarkData;
// Completes once with either a session or an error, never both. May complete
// synchronously on the calling thread.
- (void)loadFunctionNamed:(NSString *)functionName
              inputNames:(NSArray<NSString *> *)inputNames
             outputNames:(NSArray<NSString *> *)outputNames
              completion:(void (^)(
                             id<ETCoreAISession> _Nullable session,
                             NSError *_Nullable error))completion;
@end

typedef NS_ENUM(NSInteger, ETCoreAIAcquisitionStatus) {
  ETCoreAIAcquisitionStatusHit = 0,
  ETCoreAIAcquisitionStatusMiss = 1,
  ETCoreAIAcquisitionStatusError = 2,
};

// Hit: nonnull model, nil error. Miss: nil model and error, only on SDK nil.
// Error: nil model, nonnull error. Function errors never produce a Miss.
// Acquisition operations call completion exactly once off the main thread.
typedef void (^ETCoreAIAcquisitionCompletion)(
    ETCoreAIAcquisitionStatus status,
    id<ETCoreAIPreparedModel> _Nullable preparedModel,
    NSError *_Nullable error);

@protocol ETCoreAIModelLoading <NSObject>
- (void)restoreModelFromBookmark:(NSData *)bookmark
                    completion:(ETCoreAIAcquisitionCompletion)completion;
// Uses only the SDK default cache and persistent policy. Never returns Miss.
- (void)specializeModelAtURL:(NSURL *)url
                 completion:(ETCoreAIAcquisitionCompletion)completion;
// Nil error means SDK-confirmed deletion. No busy/absent classification is inferred.
- (void)evictModelWithBookmark:(NSData *)bookmark
                  completion:(void (^)(NSError *_Nullable error))completion;
@end

#ifdef __cplusplus
extern "C" {
#endif
BOOL ETCoreAIIsAvailable(void);
NSString *_Nullable ETCoreAIDeviceArchitectureName(void);
id<ETCoreAIModelLoading> ETCoreAICreateModelLoader(void);
#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
