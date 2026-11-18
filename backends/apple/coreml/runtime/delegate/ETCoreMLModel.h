//
// ETCoreMLModel.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <vector>

NS_ASSUME_NONNULL_BEGIN

@class ETCoreMLAsset;

namespace executorchcoreml {
class MultiArray;
}

/// Represents a ML model, the class is a thin wrapper over `MLModel` with additional properties.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModel` instance.
///
/// @param asset The asset from which the model is loaded.
/// @param configuration The model configuration.
/// @param orderedInputNames   The ordered input names of the model.
/// @param orderedOutputNames   The ordered output names of the model.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithAsset:(ETCoreMLAsset*)asset
                         configuration:(MLModelConfiguration*)configuration
                     orderedInputNames:(NSOrderedSet<NSString*>*)orderedInputNames
                    orderedOutputNames:(NSOrderedSet<NSString*>*)orderedOutputNames
                                 error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// Constructs an `ETCoreMLModel` instance from a compiled model URL (for cache-based loading).
///
/// @param compiledModelURL The URL of the compiled model (.mlmodelc directory).
/// @param identifier The unique identifier for this model.
/// @param configuration The model configuration.
/// @param orderedInputNames   The ordered input names of the model.
/// @param orderedOutputNames   The ordered output names of the model.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithCompiledModelURL:(NSURL*)compiledModelURL
                                       identifier:(NSString*)identifier
                                    configuration:(MLModelConfiguration*)configuration
                                orderedInputNames:(NSOrderedSet<NSString*>*)orderedInputNames
                               orderedOutputNames:(NSOrderedSet<NSString*>*)orderedOutputNames
                                            error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// The underlying MLModel.
@property (strong, readonly, nonatomic) MLModel* mlModel;

/// The model state.
@property (strong, readonly, nonatomic, nullable) id state;

/// The asset from which the model is loaded (for asset-based loading).
@property (strong, readonly, nonatomic, nullable) ETCoreMLAsset* asset;

/// The model identifier.
@property (copy, readonly, nonatomic) NSString* identifier;

/// The ordered input names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedInputNames;

/// The ordered output names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedOutputNames;


- (nullable id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)input
                                                 options:(MLPredictionOptions*)options
                                                   error:(NSError* __autoreleasing*)error;

- (nullable NSArray<MLMultiArray*>*)prepareInputs:(const std::vector<executorchcoreml::MultiArray>&)inputs
                                            error:(NSError* __autoreleasing*)error;

- (nullable NSArray<MLMultiArray*>*)prepareOutputBackings:(const std::vector<executorchcoreml::MultiArray>&)outputs
                                                    error:(NSError* __autoreleasing*)error;

- (BOOL)prewarmAndReturnError:(NSError* __autoreleasing*)error;

/// Binds an IOSurface-backed `kCVPixelFormatType_OneComponent16Half` pixel buffer to this model, so that an input or
/// output whose data pointer is the buffer's base address is passed to Core ML as the pixel buffer: Core ML reads or
/// writes it in place at the buffer's row stride instead of copying it. Must be called before the first prediction.
///
/// @param pixelBuffer The pixel buffer, retained for the model's lifetime.
/// @param error On failure, error is filled with the failure information.
/// @retval `YES` if the buffer was bound, `NO` if it is not an IOSurface-backed fp16 buffer, the OS is older than
/// macOS 12 / iOS 16, locking fails, or the buffer has no base address.
- (BOOL)bindPixelBuffer:(CVPixelBufferRef)pixelBuffer error:(NSError* __autoreleasing*)error;

/// Sets `multiArray` to an array backed by the pixel buffer bound at `arg`'s data pointer, or to `nil` if no buffer is
/// bound there. While `arg`'s shape stays the same, every call returns the same array.
///
/// @retval `NO` if the bound pixel buffer doesn't have `arg`'s shape and data type.
- (BOOL)getPixelBufferMultiArray:(MLMultiArray* _Nullable __autoreleasing* _Nonnull)multiArray
                          forArg:(const executorchcoreml::MultiArray&)arg
                           error:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
