
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraph (VenturaOps)

#if !defined(__MAC_13_0) && (!defined(MAC_OS_X_VERSION_13_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_0))

typedef NS_ENUM(NSUInteger, MPSGraphResizeNearestRoundingMode) {
  MPSGraphResizeNearestRoundingModeRoundPreferCeil = 0L,
  MPSGraphResizeNearestRoundingModeRoundPreferFloor = 1L,
  MPSGraphResizeNearestRoundingModeCeil = 2L,
  MPSGraphResizeNearestRoundingModeFloor = 3L,
  MPSGraphResizeNearestRoundingModeRoundToEven = 4L,
  MPSGraphResizeNearestRoundingModeRoundToOdd = 5L,
};

// Define complex enums for MacOS 12
#define MPSDataTypeComplexBit 0x01000000
#define MPSDataTypeComplexFloat32 ((MPSDataType)(MPSDataTypeFloatBit | MPSDataTypeComplexBit | 64))
#define MPSDataTypeComplexFloat16 ((MPSDataType)(MPSDataTypeFloatBit | MPSDataTypeComplexBit | 32))
#endif

- (MPSGraphTensor *_Nonnull)cumulativeSumWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                               axis:(NSInteger)axis
                                               name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                      axis:(NSInteger)axis
                                      name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                      axis:(NSInteger)axis
                                descending:(BOOL)descending
                                      name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                axisTensor:(MPSGraphTensor *_Nonnull)axisTensor
                                descending:(BOOL)descending
                                      name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                axisTensor:(MPSGraphTensor *_Nonnull)axisTensor
                                      name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)argSortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                         axis:(NSInteger)axis
                                         name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)argSortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                         axis:(NSInteger)axis
                                   descending:(BOOL)descending
                                         name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)argSortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                   axisTensor:(MPSGraphTensor *_Nonnull)axisTensor
                                   descending:(BOOL)descending
                                         name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)argSortWithTensor:(MPSGraphTensor *_Nonnull)tensor
                                   axisTensor:(MPSGraphTensor *_Nonnull)axisTensor
                                         name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)inverseOfTensor:(MPSGraphTensor *_Nonnull)inputTensor name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeNearestWithTensor:(MPSGraphTensor *_Nonnull)imagesTensor
                                         sizeTensor:(MPSGraphTensor *_Nonnull)size
                                nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                       centerResult:(BOOL)centerResult
                                       alignCorners:(BOOL)alignCorners
                                             layout:(MPSGraphTensorNamedDataLayout)layout
                                               name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeNearestWithTensor:(MPSGraphTensor *_Nonnull)imagesTensor
                                         sizeTensor:(MPSGraphTensor *_Nonnull)size
                                  scaleOffsetTensor:(MPSGraphTensor *_Nonnull)scaleOffset
                                nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                             layout:(MPSGraphTensorNamedDataLayout)layout
                                               name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeBilinearWithTensor:(MPSGraphTensor *_Nonnull)imagesTensor
                                          sizeTensor:(MPSGraphTensor *_Nonnull)size
                                        centerResult:(BOOL)centerResult
                                        alignCorners:(BOOL)alignCorners
                                              layout:(MPSGraphTensorNamedDataLayout)layout
                                                name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeBilinearWithTensor:(MPSGraphTensor *_Nonnull)imagesTensor
                                          sizeTensor:(MPSGraphTensor *_Nonnull)size
                                   scaleOffsetTensor:(MPSGraphTensor *_Nonnull)scaleOffset
                                              layout:(MPSGraphTensorNamedDataLayout)layout
                                                name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeNearestWithGradientTensor:(MPSGraphTensor *_Nonnull)gradient
                                                      input:(MPSGraphTensor *_Nonnull)input
                                        nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                               centerResult:(BOOL)centerResult
                                               alignCorners:(BOOL)alignCorners
                                                     layout:(MPSGraphTensorNamedDataLayout)layout
                                                       name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeNearestWithGradientTensor:(MPSGraphTensor *_Nonnull)gradient
                                                      input:(MPSGraphTensor *_Nonnull)input
                                          scaleOffsetTensor:(MPSGraphTensor *_Nonnull)scaleOffset
                                        nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                                     layout:(MPSGraphTensorNamedDataLayout)layout
                                                       name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeBilinearWithGradientTensor:(MPSGraphTensor *_Nonnull)gradient
                                                       input:(MPSGraphTensor *_Nonnull)input
                                                centerResult:(BOOL)centerResult
                                                alignCorners:(BOOL)alignCorners
                                                      layout:(MPSGraphTensorNamedDataLayout)layout
                                                        name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)resizeBilinearWithGradientTensor:(MPSGraphTensor *_Nonnull)gradient
                                                       input:(MPSGraphTensor *_Nonnull)input
                                           scaleOffsetTensor:(MPSGraphTensor *_Nonnull)scaleOffset
                                                      layout:(MPSGraphTensorNamedDataLayout)layout
                                                        name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sampleGridWithSourceTensor:(MPSGraphTensor *_Nonnull)source
                                      coordinateTensor:(MPSGraphTensor *_Nonnull)coordinates
                                                layout:(MPSGraphTensorNamedDataLayout)layout
                                  normalizeCoordinates:(BOOL)normalizeCoordinates
                                   relativeCoordinates:(BOOL)relativeCoordinates
                                          alignCorners:(BOOL)alignCorners
                                           paddingMode:(MPSGraphPaddingMode)paddingMode
                                          samplingMode:(MPSGraphResizeMode)samplingMode
                                         constantValue:(double)constantValue
                                                  name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)sampleGridWithSourceTensor:(MPSGraphTensor *_Nonnull)source
                                      coordinateTensor:(MPSGraphTensor *_Nonnull)coordinates
                                                layout:(MPSGraphTensorNamedDataLayout)layout
                                  normalizeCoordinates:(BOOL)normalizeCoordinates
                                   relativeCoordinates:(BOOL)relativeCoordinates
                                          alignCorners:(BOOL)alignCorners
                                           paddingMode:(MPSGraphPaddingMode)paddingMode
                                   nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                         constantValue:(double)constantValue
                                                  name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)truncateWithTensor:(MPSGraphTensor *_Nonnull)tensor name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)transposeTensor:(MPSGraphTensor *_Nonnull)tensor
                                permutation:(NSArray<NSNumber *> *_Nonnull)permutation
                                       name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)bitwiseANDWithPrimaryTensor:(MPSGraphTensor *_Nonnull)primaryTensor
                                        secondaryTensor:(MPSGraphTensor *_Nonnull)secondaryTensor
                                                   name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)bitwiseORWithPrimaryTensor:(MPSGraphTensor *_Nonnull)primaryTensor
                                       secondaryTensor:(MPSGraphTensor *_Nonnull)secondaryTensor
                                                  name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)bitwiseXORWithPrimaryTensor:(MPSGraphTensor *_Nonnull)primaryTensor
                                        secondaryTensor:(MPSGraphTensor *_Nonnull)secondaryTensor
                                                   name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nonnull)bitwiseNOTWithTensor:(MPSGraphTensor *_Nonnull)tensor name:(NSString *_Nullable)name;

#if !defined(MAC_OS_X_VERSION_12_2) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_12_2)
- (MPSGraphTensor *_Nullable)expandDimsOfTensor:(MPSGraphTensor *_Nullable)tensor
                                           axis:(NSInteger)axis
                                           name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nullable)expandDimsOfTensor:(MPSGraphTensor *_Nullable)tensor
                                           axes:(NSArray<NSNumber *> *_Nullable)axes
                                           name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nullable)squeezeTensor:(MPSGraphTensor *_Nullable)tensor
                                      axes:(NSArray<NSNumber *> *_Nullable)axes
                                      name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nullable)squeezeTensor:(MPSGraphTensor *_Nullable)tensor
                                      axis:(NSInteger)axis
                                      name:(NSString *_Nullable)name;

- (NSArray<MPSGraphTensor *> *_Nullable)
    maxPooling2DReturnIndicesWithSourceTensor:(MPSGraphTensor *_Nullable)source
                                   descriptor:(MPSGraphPooling2DOpDescriptor *_Nullable)descriptor
                                         name:(NSString *_Nullable)name;

- (MPSGraphTensor *_Nullable)coordinateAlongAxis:(NSInteger)axis
                                 withShapeTensor:(MPSGraphTensor *_Nullable)shapeTensor
                                            name:(NSString *_Nullable)name;

- (NSArray<MPSGraphTensor *> *_Nullable)splitTensor:(MPSGraphTensor *_Nullable)tensor
                                   splitSizesTensor:(MPSGraphTensor *_Nullable)splitSizesTensor
                                               axis:(NSInteger)axis
                                               name:(NSString *_Nullable)name;

- (NSArray<MPSGraphTensor *> *_Nullable)splitTensor:(MPSGraphTensor *_Nullable)tensor
                                         splitSizes:(NSArray<NSNumber *> *_Nullable)splitSizes
                                               axis:(NSInteger)axis
                                               name:(NSString *_Nullable)name;
#endif

@end
