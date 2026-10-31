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
  ETCoreAIScalarTypeInt32 = 2,
  ETCoreAIScalarTypeBool = 3,
};

// One value per Core AI SpecializationOptions configuration: the default (all
// compute units), cpuOnly, or a preferred compute unit kind. Preferences still
// allow every unit for operations the preferred unit cannot run.
typedef NS_ENUM(NSInteger, ETCoreAIComputeUnit) {
  ETCoreAIComputeUnitDefault = 0,
  ETCoreAIComputeUnitCPUOnly = 1,
  ETCoreAIComputeUnitPreferCPU = 2,
  ETCoreAIComputeUnitPreferGPU = 3,
  ETCoreAIComputeUnitPreferNeuralEngine = 4,
};

typedef NS_ENUM(NSInteger, ETCoreAIErrorCode) {
  // Invalid model descriptors, functions or bindings.
  ETCoreAIErrorInvalidModel = 1,
  // Unsupported tensor types, layouts or model features.
  ETCoreAIErrorUnsupported = 2,
  // Invalid caller arguments or input tensor payloads.
  ETCoreAIErrorInvalidArgument = 3,
  // SDK or system failures, or inconsistent SDK outputs. Does not mean cached
  // files are corrupt.
  ETCoreAIErrorRuntime = 4,
};

FOUNDATION_EXPORT NSErrorDomain const ETCoreAIErrorDomain;

// Input storage is borrowed until output copying finishes. The bridge never
// retains or frees it.
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

// Holds the SDK output until copyBytesTo: runs; raw SDK pointers are used only
// inside that call. Construct only after validating the source layout, because
// copying cannot report errors.
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

// Keep inputs valid until output copying finishes or completion reports
// failure. Outputs may share input storage. Stage copies before writing any
// destination that aliases an input or another output destination.
//
// Completion runs exactly once, possibly before executeInputs returns, on an
// unspecified thread or queue.
@protocol ETCoreAISession<NSObject>
- (void)executeInputs:(NSArray<ETCoreAIInputTensor*>*)inputs
           completion:(void (^)(
                          NSArray<ETCoreAITensor*>* _Nullable outputs,
                          NSError* _Nullable error))completion;
@end

// Holds the SDK model, keeping its cache entry in use. Bound sessions hold the
// model too, so they stay valid after the prepared model is released.
@protocol ETCoreAIPreparedModel<NSObject>
- (NSData*)copyBookmarkData;
// Completes once with either a session or an error, never both. Completion may
// run synchronously on the calling thread.
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

// Hit: nonnull model, nil error.
// Miss: nil model and error, only for an SDK nil result.
// Error: nil model, nonnull error. Function errors never report Miss.
//
// Acquisition completions run exactly once off the main thread.
typedef void (^ETCoreAIAcquisitionCompletion)(
    ETCoreAIAcquisitionStatus status,
    id<ETCoreAIPreparedModel> _Nullable preparedModel,
    NSError* _Nullable error);

@protocol ETCoreAIModelLoading<NSObject>
// Stable for this loader's lifetime; derived from its specialization options.
@property(nonatomic, readonly, copy) NSString* specializationIdentifier;
- (void)restoreModelFromBookmark:(NSData*)bookmark
                      completion:(ETCoreAIAcquisitionCompletion)completion;
// Uses the SDK default cache with storage-pressure-only purging.
// Specialization never reports Miss.
- (void)specializeModelAtURL:(NSURL*)url
                  completion:(ETCoreAIAcquisitionCompletion)completion;
// Nil error means SDK-confirmed deletion; the bridge adds no
// busy/absent classification.
- (void)evictModelWithBookmark:(NSData*)bookmark
                    completion:(void (^)(NSError* _Nullable error))completion;
@end

#ifdef __cplusplus
extern "C" {
#endif
BOOL ETCoreAIIsAvailable(void);
NSString* _Nullable ETCoreAIDeviceArchitectureName(void);
id<ETCoreAIModelLoading> ETCoreAICreateModelLoader(
    ETCoreAIComputeUnit computeUnit);
#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
