//
// ETCoreMLOperationProfilingInfo.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLOperationProfilingInfo.h"

#import "hash_util.h"
#import "model_event_logger_impl.h"

namespace  {
NSString *const kPreferredComputeUnitKey = @"preferredComputeUnit";
NSString *const kSupportedComputeUnitsKey = @"supportedComputeUnits";
NSString *const kOutputNamesKey = @"outputNames";
NSString *const kOperatorNameKey = @"operatorName";
NSString *const kEstimatedCostKey = @"estimatedCost";

NSArray<NSString *> *compute_unit_names(ETCoreMLComputeUnits compute_units) {
    NSMutableArray<NSString *> *result = [NSMutableArray array];
    if ((compute_units & ETCoreMLComputeUnitNeuralEngine) > 0) {
        [result addObject:@"NE"];
    }
    if ((compute_units & ETCoreMLComputeUnitCPU) > 0) {
        [result addObject:@"CPU"];
    }
    if ((compute_units & ETCoreMLComputeUnitGPU) > 0) {
        [result addObject:@"GPU"];
    }
    
    return result;
}

NSData *get_metadata(ETCoreMLComputeUnits preferredComputeUnit,
                     ETCoreMLComputeUnits supportedComputeUnits,
                     NSArray<NSString *> *outputNames,
                     NSString *operatorName,
                     double estimatedCost) {
    NSMutableDictionary<NSString *, id> *result = [NSMutableDictionary new];
    result[kPreferredComputeUnitKey] = compute_unit_names(preferredComputeUnit).firstObject;
    result[kSupportedComputeUnitsKey] = compute_unit_names(supportedComputeUnits);
    result[kOutputNamesKey] = outputNames;
    result[kOperatorNameKey] = operatorName;
    result[kEstimatedCostKey] = @(estimatedCost);
    NSError *local_error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingOptions(0) error:&local_error];
    NSCAssert(data != nil, @"%@: Failed to serialize metadata.", NSStringFromClass(ETCoreMLOperationProfilingInfo.class));
    
    return data;
};
}

@implementation ETCoreMLOperationProfilingInfo

- (instancetype)initWithPreferredComputeUnit:(ETCoreMLComputeUnits)preferredComputeUnit
                       supportedComputeUnits:(ETCoreMLComputeUnits)supportedComputeUnits
                 estimatedExecutionStartTime:(uint64_t)estimatedExecutionStartTime
                   estimatedExecutionEndTime:(uint64_t)estimatedExecutionEndTime
                               estimatedCost:(double)estimatedCost
                                 outputNames:(NSArray<NSString *> *)outputNames
                                operatorName:(NSString *)operatorName
                                    metadata:(NSData *)metadata {
    self = [super init];
    if (self) {
        _preferredComputeUnit = preferredComputeUnit;
        _supportedComputeUnits = supportedComputeUnits;
        _estimatedExecutionStartTime = estimatedExecutionStartTime;
        _estimatedExecutionEndTime = estimatedExecutionEndTime;
        _estimatedCost = estimatedCost;
        _outputNames = [outputNames copy];
        _operatorName = [operatorName copy];
        _metadata = get_metadata(preferredComputeUnit, supportedComputeUnits, outputNames, operatorName, estimatedCost);
    }
    
    return self;
}

- (instancetype)initWithPreferredComputeUnit:(ETCoreMLComputeUnits)preferredComputeUnit
                       supportedComputeUnits:(ETCoreMLComputeUnits)supportedComputeUnits
                 estimatedExecutionStartTime:(uint64_t)estimatedExecutionStartTime
                   estimatedExecutionEndTime:(uint64_t)estimatedExecutionEndTime
                               estimatedCost:(double)estimatedCost
                                 outputNames:(NSArray<NSString *> *)outputNames
                                operatorName:(NSString *)operatorName {
    NSData *metadata = get_metadata(preferredComputeUnit, supportedComputeUnits, outputNames, operatorName, estimatedCost);
    return [self initWithPreferredComputeUnit:preferredComputeUnit
                        supportedComputeUnits:supportedComputeUnits
                  estimatedExecutionStartTime:estimatedExecutionStartTime
                    estimatedExecutionEndTime:estimatedExecutionEndTime
                                estimatedCost:estimatedCost
                                  outputNames:outputNames
                                 operatorName:operatorName
                                     metadata:metadata];
}

- (BOOL)isEqual:(id)object {
    if (object == self) {
        return YES;
    }
    
    if (![object isKindOfClass:self.class]) {
        return NO;
    }
    
    ETCoreMLOperationProfilingInfo *other = (ETCoreMLOperationProfilingInfo *)object;
    
    return self.preferredComputeUnit == other.preferredComputeUnit &&
    self.supportedComputeUnits == other.supportedComputeUnits &&
    self.estimatedExecutionStartTime == other.estimatedExecutionStartTime &&
    self.estimatedExecutionEndTime == other.estimatedExecutionEndTime &&
    [self.outputNames isEqualToArray:other.outputNames] &&
    [self.operatorName isEqualToString:other.operatorName] &&
    [self.metadata isEqualToData:other.metadata];
}

- (NSUInteger)hash {
    size_t seed = 0;
    executorchcoreml::hash_combine(seed, self.preferredComputeUnit);
    executorchcoreml::hash_combine(seed, self.supportedComputeUnits);
    executorchcoreml::hash_combine(seed, self.estimatedExecutionStartTime);
    executorchcoreml::hash_combine(seed, self.estimatedExecutionEndTime);
    executorchcoreml::hash_combine(seed, self.outputNames.hash);
    executorchcoreml::hash_combine(seed, self.operatorName.hash);
    executorchcoreml::hash_combine(seed, self.metadata.hash);
    
    return seed;
}

- (instancetype)copyWithZone:(NSZone *)zone {
    return [[ETCoreMLOperationProfilingInfo allocWithZone:zone] initWithPreferredComputeUnit:self.preferredComputeUnit
                                                                       supportedComputeUnits:self.supportedComputeUnits
                                                                 estimatedExecutionStartTime:self.estimatedExecutionStartTime
                                                                   estimatedExecutionEndTime:self.estimatedExecutionEndTime
                                                                               estimatedCost:self.estimatedCost
                                                                                 outputNames:self.outputNames
                                                                                operatorName:self.operatorName
                                                                                    metadata:self.metadata];
}

@end
