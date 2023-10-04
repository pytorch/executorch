//
// ETCoreMLModel.h
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@class ETCoreMLAsset;

/// Represents a ML model, the class is a thin wrapper over `MLModel` with additional properties.
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

/// The underlying MLModel.
@property (strong, readonly, nonatomic) MLModel* mlModel;

/// The asset from which the model is loaded.
@property (strong, readonly, nonatomic) ETCoreMLAsset* asset;

/// The asset identifier.
@property (strong, readonly, nonatomic) NSString* identifier;

/// The ordered input names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedInputNames;

/// The ordered output names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedOutputNames;

@end

NS_ASSUME_NONNULL_END
