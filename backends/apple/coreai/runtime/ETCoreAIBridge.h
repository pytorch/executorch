/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Bridge completions can require Swift's cooperative thread pool. Callers that
// synchronously wait for them must use ordinary worker threads, not Swift Task
// or Task.detached workers, which can starve the work needed for completion.

typedef NS_ENUM(NSInteger, ETCoreAIScalarType) {
  ETCoreAIScalarTypeFloat16 = 0,
  ETCoreAIScalarTypeFloat32 = 1,
};

typedef NS_ENUM(NSInteger, ETCoreAIErrorCode) {
  // Invalid model descriptors, functions or bindings.
  ETCoreAIErrorInvalidModel = 1,
  // Unsupported tensor types, layouts or model features.
  ETCoreAIErrorUnsupported = 2,
  // Invalid caller arguments or input tensor payloads.
  ETCoreAIErrorInvalidArgument = 3,
  // SDK/system failures or inconsistent SDK outputs, not evidence of
  // corruption.
  ETCoreAIErrorRuntime = 4,
};

FOUNDATION_EXPORT NSErrorDomain const ETCoreAIErrorDomain;

// Input storage is borrowed through output copying, never retained or freed.
@interface ETCoreAIInputTensor : NSObject
@property(nonatomic, readonly) const void* _Nullable bytes;
@property(nonatomic, readonly) NSUInteger byteCount;
@property(nonatomic, readonly, copy) NSArray<NSNumber*>* shape;
@property(nonatomic, readonly) ETCoreAIScalarType scalarType;
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)initWithBytes:(const void* _Nullable)bytes
                    byteCount:(NSUInteger)byteCount
                        shape:(NSArray<NSNumber*>*)shape
                   scalarType:(ETCoreAIScalarType)scalarType
    NS_DESIGNATED_INITIALIZER;
@end

// The copier retains its source storage; SDK pointers never escape their access
// scope. Construct only after validating the source layout. Copying cannot
// fail.
@interface ETCoreAITensor : NSObject
@property(nonatomic, readonly) NSUInteger byteCount;
@property(nonatomic, readonly, copy) NSArray<NSNumber*>* shape;
@property(nonatomic, readonly) ETCoreAIScalarType scalarType;
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)initWithByteCount:(NSUInteger)byteCount
                            shape:(NSArray<NSNumber*>*)shape
                       scalarType:(ETCoreAIScalarType)scalarType
                        copyBytes:(void (^)(void* bytes))copyBytes
    NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithData:(NSData*)data
                       shape:(NSArray<NSNumber*>*)shape
                  scalarType:(ETCoreAIScalarType)scalarType;
// Caller supplies byteCount writable bytes, disjoint from live source storage.
// A null destination is allowed only for an empty output.
- (void)copyBytesTo:(void* _Nullable)bytes NS_SWIFT_NAME(copyBytes(to:));
@end

// The caller keeps inputs valid until all output copies finish, or completion
// reports failure. Outputs may share input storage; stage copies before writing
// any destination that aliases an input or another output destination.
// Completion is called exactly once, possibly before executeInputs returns,
// with no guarantee of a particular thread or queue.
@protocol ETCoreAISession<NSObject>
- (void)executeInputs:(NSArray<ETCoreAIInputTensor*>*)inputs
           completion:(void (^)(
                          NSArray<ETCoreAITensor*>* _Nullable outputs,
                          NSError* _Nullable error))completion;
@end

// A prepared model pins the SDK entry. Sessions retain their own pin after
// binding.
@protocol ETCoreAIPreparedModel<NSObject>
- (NSData*)copyBookmarkData;
// Completes once with either a session or an error, never both. May complete
// synchronously on the calling thread.
- (void)loadFunctionNamed:(NSString*)functionName
               inputNames:(NSArray<NSString*>*)inputNames
              outputNames:(NSArray<NSString*>*)outputNames
               completion:(void (^)(
                              id<ETCoreAISession> _Nullable session,
                              NSError* _Nullable error))completion;
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
    NSError* _Nullable error);

@protocol ETCoreAIModelLoading<NSObject>
// Stable for this loader's lifetime; derived from its actual specialization
// options.
@property(nonatomic, readonly, copy) NSString* specializationIdentifier;
- (void)restoreModelFromBookmark:(NSData*)bookmark
                      completion:(ETCoreAIAcquisitionCompletion)completion;
// Uses the SDK default cache with storage-pressure-only purging. Never returns
// Miss.
- (void)specializeModelAtURL:(NSURL*)url
                  completion:(ETCoreAIAcquisitionCompletion)completion;
// Nil error means SDK-confirmed deletion. No busy/absent classification is
// inferred.
- (void)evictModelWithBookmark:(NSData*)bookmark
                    completion:(void (^)(NSError* _Nullable error))completion;
@end

#ifdef __cplusplus
extern "C" {
#endif
BOOL ETCoreAIIsAvailable(void);
NSString* _Nullable ETCoreAIDeviceArchitectureName(void);
id<ETCoreAIModelLoading> ETCoreAICreateModelLoader(void);
#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
