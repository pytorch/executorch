//
// ETCoreMLTestUtils.h
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelAnalyzer.h>

NS_ASSUME_NONNULL_BEGIN

@interface ETCoreMLTestUtils : NSObject

/// Creates a unique directory inside the specified directory.
///
/// The created directory contains a nested directory named 'folder' and a 'content.txt' file inside
/// 'folder' directory with the specified content.
///
/// @param directoryURL The url of the directory.
/// @param content   The content of 'content.txt' file.
/// @param fileManager The file manager.
/// @param error   On failure, error is filled with the failure information.
/// @retval The directory URL if the creation succeeded otherwise `nil`.
+ (nullable NSURL*)createUniqueAssetInDirectoryAtURL:(NSURL*)directoryURL
                                         withContent:(NSString*)content
                                         fileManager:(NSFileManager*)fileManager
                                               error:(NSError* __autoreleasing*)error;


/// Creates a `ETCoreMLAssetManager` for testing.
///
///
/// @param directoryURL The url of the directory.
/// @param error   On failure, error is filled with the failure information.
/// @retval An `ETCoreMLAssetManager` instance if the creation succeeded otherwise `nil`.
+ (nullable ETCoreMLAssetManager*)createAssetManagerWithURL:(NSURL*)directoryURL error:(NSError* __autoreleasing*)error;


/// Creates a `MLMultiArray` instance with a repeated value.
///
/// @param constraint The MLMultiArray constraint.
/// @param value The value to be repeated.
/// @param error   On failure, error is filled with the failure information.
/// @retval A  filled `MLMultiArray` instance.
+ (nullable MLMultiArray*)filledMultiArrayWithConstraint:(MLMultiArrayConstraint*)constraint
                                           repeatedValue:(NSNumber*)value
                                                   error:(NSError* __autoreleasing*)error;


/// Creates a `MLMultiArray` instance with a repeated value.
///
/// @param shape The MLMultiArray shape.
/// @param dataType The MLMultiArray data type.
/// @param error   On failure, error is filled with the failure information.
/// @retval A  filled `MLMultiArray` instance.
+ (nullable MLMultiArray*)filledMultiArrayWithShape:(NSArray<NSNumber*>*)shape
                                           dataType:(MLMultiArrayDataType)dataType
                                      repeatedValue:(NSNumber*)value
                                              error:(NSError* __autoreleasing*)error;


/// Creates inputs with repeated values for a `ETCoreMLModel`.
///
/// @param model The model.
/// @param repeatedValues An array of values, the size of the array must be equal to the inputs count.
/// @param error   On failure, error is filled with the failure information.
/// @retval Model inputs with repeated values.
+ (nullable NSArray<MLMultiArray*>*)inputsForModel:(ETCoreMLModel*)model
                                    repeatedValues:(NSArray<NSNumber*>*)repeatedValues
                                             error:(NSError* __autoreleasing*)error;

/// Creates input features with repeated values for a `ETCoreMLModel`.
///
/// @param model The model.
/// @param repeatedValues An array of values, the size of the array must be equal to the inputs count.
/// @param error   On failure, error is filled with the failure information.
/// @retval Model inputs with repeated values.
+ (nullable id<MLFeatureProvider>)inputFeaturesForModel:(ETCoreMLModel*)model
                                         repeatedValues:(NSArray<NSNumber*>*)repeatedValues
                                                  error:(NSError* __autoreleasing*)error;

/// Creates a `ETCoreMLModelAnalyzer`instance for analyzing (debugging and profiling) a CoreML model.
///
/// @param data The AOT data.
/// @param dstURL The folder url that will be used for managing the assets.
/// @param error   On failure, error is filled with the failure information.
/// @retval An `ETCoreMLModelAnalyzer` instance if the creation succeeded otherwise `nil`.
+ (nullable ETCoreMLModelAnalyzer*)createAnalyzerWithAOTData:(NSData*)data
                                                      dstURL:(NSURL*)dstURL
                                                       error:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
