//
// ETCoreMLOperationProfilingInfo.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

#import <ETCoreMLComputeUnits.h>

NS_ASSUME_NONNULL_BEGIN

/// A class representing the profiling info of an operation.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLOperationProfilingInfo : NSObject<NSCopying>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLOperationProfilingInfo` instance.
///
/// @param preferredComputeUnit The compute unit used to execute the operation.
/// @param supportedComputeUnits The compute units that can execute the operation.
/// @param estimatedExecutionStartTime The estimated execution start time.
/// @param estimatedExecutionEndTime The estimated execution end time.
/// @param estimatedCost The estimated cost of executing an operation.
/// @param outputNames The output names of the operation.
/// @param operatorName The operator name of the operation.
/// @param metadata The metadata, for logging additional info.
- (instancetype)initWithPreferredComputeUnit:(ETCoreMLComputeUnits)preferredComputeUnit
                       supportedComputeUnits:(ETCoreMLComputeUnits)supportedComputeUnits
                 estimatedExecutionStartTime:(uint64_t)estimatedExecutionStartTime
                   estimatedExecutionEndTime:(uint64_t)estimatedExecutionEndTime
                               estimatedCost:(double)estimatedCost
                                 outputNames:(NSArray<NSString*>*)outputNames
                                operatorName:(NSString*)operatorName
                                    metadata:(NSData*)metadata NS_DESIGNATED_INITIALIZER;

/// Constructs an `ETCoreMLOperationProfilingInfo` instance.
///
/// @param preferredComputeUnit The compute unit used to execute the operation.
/// @param supportedComputeUnits The compute units that can execute the operation.
/// @param estimatedExecutionStartTime The estimated execution start time.
/// @param estimatedExecutionEndTime The estimated execution end time.
/// @param estimatedCost The estimated cost of executing an operation.
/// @param outputNames The output names of the operation.
/// @param operatorName The operator name of the operation.
- (instancetype)initWithPreferredComputeUnit:(ETCoreMLComputeUnits)preferredComputeUnit
                       supportedComputeUnits:(ETCoreMLComputeUnits)supportedComputeUnits
                 estimatedExecutionStartTime:(uint64_t)estimatedExecutionStartTime
                   estimatedExecutionEndTime:(uint64_t)estimatedExecutionEndTime
                               estimatedCost:(double)estimatedCost
                                 outputNames:(NSArray<NSString*>*)outputNames
                                operatorName:(NSString*)operatorName;
/// The preferred compute unit.
@property (readonly, assign, nonatomic) ETCoreMLComputeUnits preferredComputeUnit;
/// The supported compute units.
@property (readonly, assign, nonatomic) ETCoreMLComputeUnits supportedComputeUnits;
/// The estimated execution start time.
@property (readwrite, assign, nonatomic) uint64_t estimatedExecutionStartTime;
/// The estimated execution end time.
@property (readwrite, assign, nonatomic) uint64_t estimatedExecutionEndTime;
/// The output names of the operation.
@property (readonly, copy, nonatomic) NSArray<NSString*>* outputNames;
/// The operator name of the operation.
@property (readonly, copy, nonatomic) NSString* operatorName;
/// The estimated cost for executing the operation.
@property (readonly, assign, nonatomic) double estimatedCost;
/// The metadata, this is used to log additional info.
@property (readonly, strong, nonatomic) NSData* metadata;

@end

NS_ASSUME_NONNULL_END
