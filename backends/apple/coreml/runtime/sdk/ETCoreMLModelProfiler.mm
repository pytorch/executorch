//
// ETCoreMLModelProfiler.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelProfiler.h"

#import "ETCoreMLAsset.h"
#import "ETCoreMLModel.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModelStructurePath.h"
#import "ETCoreMLOperationProfilingInfo.h"
#import "ETCoreMLPair.h"
#import "ETCoreMLStrings.h"
#import <mach/mach_time.h>
#import <math.h>
#import "program_path.h"

namespace  {
using namespace executorchcoreml::modelstructure;

#if MODEL_PROFILING_IS_AVAILABLE

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
MLComputePlan *_Nullable get_compute_plan_of_model_at_url(NSURL *model_url,
                                                          MLModelConfiguration *configuration,
                                                          NSError* __autoreleasing *error) {
    __block NSError *local_error = nil;
    __block MLComputePlan *result = nil;
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:model_url configuration:configuration completionHandler:^(MLComputePlan * _Nullable compute_plan,
                                                                                               NSError * _Nullable compute_plan_error) {
        result = compute_plan;
        local_error = compute_plan_error;
        dispatch_semaphore_signal(sema);
    }];
    
    long status = dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * 60 * NSEC_PER_SEC)));
    if (status != 0) {
        ETCoreMLLogUnderlyingErrorAndSetNSError(error,
                                                ETCoreMLErrorCompilationFailed,
                                                local_error,
                                                "%@: Failed to get compute plan of model with name=%@.",
                                                NSStringFromClass(ETCoreMLModelProfiler.class),
                                                model_url.lastPathComponent);
        return nil;
    }
    
    return result;
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
void visit_program_operation(MLModelStructureProgramBlock *block,
                             const Path& block_path,
                             BOOL (^handler)(MLModelStructureProgramOperation *operation, ETCoreMLModelStructurePath *path)) {
    for (MLModelStructureProgramOperation *operation in block.operations) {
        Path operation_path = block_path;
        operation_path.append_component(Path::Program::Operation(operation.outputs.firstObject.name.UTF8String));
        if (!handler(operation, [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:operation_path])) {
            return;
        }
        
        for (size_t i = 0; i < operation.blocks.count; ++i) {
            Path nested_block_path = operation_path;
            nested_block_path.append_component(Path::Program::Block(i));
            visit_program_operation(operation.blocks[i], nested_block_path,handler);
        }
    }
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
void visit_program_operation(MLModelStructure *modelStructure, BOOL (^handler)(MLModelStructureProgramOperation *operation, ETCoreMLModelStructurePath *path)) {
    using namespace executorchcoreml::modelstructure;
    [modelStructure.program.functions enumerateKeysAndObjectsUsingBlock:^(NSString *function_name,
                                                                          MLModelStructureProgramFunction *function,
                                                                          BOOL * _Nonnull __unused stop) {
        Path path;
        path.append_component(Path::Program());
        path.append_component(Path::Program::Function(function_name.UTF8String));
        path.append_component(Path::Program::Block(-1));
        visit_program_operation(function.block, path, handler);
    }];
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
ETCoreMLComputeUnits to_compute_unit(id<MLComputeDeviceProtocol> compute_device) {
    if ([compute_device isKindOfClass:MLCPUComputeDevice.class]) {
        return ETCoreMLComputeUnitCPU;
    } else if ([compute_device isKindOfClass:MLGPUComputeDevice.class]) {
        return ETCoreMLComputeUnitGPU;
    } else if ([compute_device isKindOfClass:MLNeuralEngineComputeDevice.class]) {
        return ETCoreMLComputeUnitNeuralEngine;
    } else {
        return ETCoreMLComputeUnitUnknown;
    }
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
ETCoreMLComputeUnits to_compute_units(NSArray<id<MLComputeDeviceProtocol>> *compute_devices) {
    ETCoreMLComputeUnits units = ETCoreMLComputeUnitUnknown;
    for (id<MLComputeDeviceProtocol> compute_device in compute_devices) {
        units |= to_compute_unit(compute_device);
    }
    
    return units;
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
ETCoreMLOperationProfilingInfo *get_profiling_info(MLComputePlanDeviceUsage *device_usage,
                                                   MLModelStructureProgramOperation *operation,
                                                   uint64_t op_execution_start_time,
                                                   uint64_t op_execution_end_time,
                                                   double estimatedCost) {
    NSMutableArray<NSString *> *outputNames = [[NSMutableArray alloc] initWithCapacity:operation.outputs.count];
    for (MLModelStructureProgramNamedValueType *output in operation.outputs) {
        [outputNames addObject:output.name];
    }
    
    ETCoreMLComputeUnits preferred_compute_unit = to_compute_unit(device_usage.preferredComputeDevice);
    ETCoreMLComputeUnits supported_compute_units = to_compute_units(device_usage.supportedComputeDevices);
    ETCoreMLOperationProfilingInfo *info = [[ETCoreMLOperationProfilingInfo alloc] initWithPreferredComputeUnit:preferred_compute_unit
                                                                                          supportedComputeUnits:supported_compute_units
                                                                                    estimatedExecutionStartTime:op_execution_start_time
                                                                                      estimatedExecutionEndTime:op_execution_end_time
                                                                                                  estimatedCost:estimatedCost
                                                                                                    outputNames:outputNames
                                                                                                   operatorName:operation.operatorName];
    return info;
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *
prepare_profiling_infos(NSArray<MLModelStructureProgramOperation *> *operations,
                        NSDictionary<NSValue *, ETCoreMLModelStructurePath *> *operation_to_path_map,
                        MLComputePlan *compute_plan) {
    NSMutableDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *result = [NSMutableDictionary dictionaryWithCapacity:operation_to_path_map.count];
    for (MLModelStructureProgramOperation *operation in operations) {
        MLComputePlanCost *estimated_cost = [compute_plan estimatedCostOfMLProgramOperation:operation];
        if (!estimated_cost || std::isnan(estimated_cost.weight)) {
            continue;
        }
        
        NSValue *key = [NSValue valueWithPointer:(const void*)operation];
        ETCoreMLModelStructurePath *path = operation_to_path_map[key];
        MLComputePlanDeviceUsage *device_usage = [compute_plan computeDeviceUsageForMLProgramOperation:operation];
        if (path && device_usage) {
            ETCoreMLOperationProfilingInfo *profiling_info = get_profiling_info(device_usage,
                                                                                operation,
                                                                                0,
                                                                                0,
                                                                                estimated_cost.weight);
            result[path] = profiling_info;
        }
    }
    
    return result;
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *
get_profiling_infos_for_paths(NSSet<ETCoreMLModelStructurePath *> *paths,
                              NSArray<MLModelStructureProgramOperation *> *topologically_sorted_operations,
                              NSDictionary<NSValue *, ETCoreMLModelStructurePath *> *operation_to_path_map,
                              NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profiling_infos,
                              uint64_t model_execution_start_time,
                              uint64_t model_execution_end_time) {
    NSMutableDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *result = [NSMutableDictionary dictionaryWithCapacity:paths.count];
    uint64_t op_execution_start_time = model_execution_start_time;
    uint64_t op_execution_end_time = model_execution_start_time;
    // `model_execution_end_time` >= `model_execution_start_time`.
    uint64_t model_execution_duration = model_execution_end_time - model_execution_start_time;
    for (MLModelStructureProgramOperation *operation in topologically_sorted_operations) {
        NSValue *key = [NSValue valueWithPointer:(const void*)operation];
        ETCoreMLModelStructurePath *path = operation_to_path_map[key];
        ETCoreMLOperationProfilingInfo *profiling_info = profiling_infos[path];
        if (!profiling_info) {
            continue;
        }
        
        op_execution_end_time = op_execution_start_time + static_cast<uint64_t>(static_cast<double>(model_execution_duration) * profiling_info.estimatedCost);
        if (path && [paths containsObject:path]) {
            ETCoreMLOperationProfilingInfo *profiling_info_new = [[ETCoreMLOperationProfilingInfo alloc] initWithPreferredComputeUnit:profiling_info.preferredComputeUnit
                                                                                                                supportedComputeUnits:profiling_info.supportedComputeUnits
                                                                                                          estimatedExecutionStartTime:op_execution_start_time
                                                                                                            estimatedExecutionEndTime:op_execution_end_time
                                                                                                                        estimatedCost:profiling_info.estimatedCost
                                                                                                                          outputNames:profiling_info.outputNames
                                                                                                                         operatorName:profiling_info.operatorName
                                                                                                                             metadata:profiling_info.metadata];
            result[path] = profiling_info_new;
        }
        op_execution_start_time = op_execution_end_time;
    }
    
    return result;
}

API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4))
BOOL is_const_operation(MLModelStructureProgramOperation *operation) {
    return [operation.operatorName isEqualToString:@"const"];
}

void set_model_outputs(id<MLFeatureProvider> output_features,
                       NSOrderedSet<NSString *> *output_names,
                       NSArray<MLMultiArray *> *_Nullable __autoreleasing *_Nonnull model_outputs) {
    NSMutableArray<MLMultiArray *> *values = [NSMutableArray arrayWithCapacity:output_names.count];
    for (NSString *output_name in output_names) {
        MLFeatureValue *feature_value = [output_features featureValueForName:output_name];
        NSCAssert(feature_value.multiArrayValue != nil, @"%@: Expected a multiarray value for output name=%@.",
                  NSStringFromClass(ETCoreMLModelProfiler.class),
                  output_name);
        [values addObject:feature_value.multiArrayValue];
    }
    
    *model_outputs = values;
}

#endif

}

@interface ETCoreMLModelProfiler ()
/// The model.
@property (readonly, strong, nonatomic) ETCoreMLModel *model;
/// The model output names.
@property (readonly, copy, nonatomic) NSOrderedSet<NSString *> *outputNames;
#if MODEL_PROFILING_IS_AVAILABLE
/// The compute plan.
@property (readonly, strong, nonatomic) MLComputePlan *computePlan API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4));
/// The topologically sorted operations.
@property (readonly, copy, nonatomic) NSArray<MLModelStructureProgramOperation *> *topologicallySortedOperations API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4), watchos(10.4));
#endif
/// The mapping from operation to it's path in the model structure.
@property (readonly, strong, nonatomic) NSDictionary<NSValue *, ETCoreMLModelStructurePath *> *operationToPathMap;
/// The profiling infos for all the operations.
@property (readonly, copy, nonatomic) NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profilingInfos;

@end

@implementation ETCoreMLModelProfiler

- (nullable instancetype)initWithModel:(ETCoreMLModel *)model
                         configuration:(MLModelConfiguration *)configuration
                                 error:(NSError * __autoreleasing *)error  {
#if MODEL_PROFILING_IS_AVAILABLE
    if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *)) {
        NSURL *compiledModelURL = model.asset.contentURL;
        MLComputePlan *computePlan = get_compute_plan_of_model_at_url(compiledModelURL,
                                                                      configuration,
                                                                      error);
        if (!computePlan) {
            return nil;
        }

        __block NSMutableArray<ETCoreMLModelStructurePath *> *operationPaths = [NSMutableArray array];
        __block NSMutableDictionary<NSValue *, ETCoreMLModelStructurePath *> *operationToPathMap = [NSMutableDictionary dictionary];
        __block NSMutableArray<MLModelStructureProgramOperation *> *topologicallySortedOperations = [NSMutableArray new];
        visit_program_operation(computePlan.modelStructure, ^BOOL(MLModelStructureProgramOperation *operation, ETCoreMLModelStructurePath *operationPath) {
            if (is_const_operation(operation)) {
                return YES;
            }
            
            [topologicallySortedOperations addObject:operation];
            NSValue *key = [NSValue valueWithPointer:(const void*)operation];
            operationToPathMap[key] = operationPath;
            [operationPaths addObject:operationPath];
            return YES;
        });
        
        NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profilingInfos = prepare_profiling_infos(topologicallySortedOperations,
                                                                                                                               operationToPathMap,
                                                                                                                               computePlan);
        
        self = [super init];
        if (self) {
            _model = model;
            _computePlan = computePlan;
            _operationToPathMap = operationToPathMap;
            _topologicallySortedOperations = topologicallySortedOperations;
            _operationPaths = operationPaths;
            _profilingInfos = profilingInfos;
        }
        
        return self;
    }
#endif
    ETCoreMLLogErrorAndSetNSError(error,
                                  ETCoreMLErrorModelProfilingNotSupported,
                                  "%@: Model profiling is only available for macOS >= 14.4, iOS >= 17.4, tvOS >= 17.4 and watchOS >= 10.4.",
                                  NSStringFromClass(self.class));
    return nil;
}

- (nullable ETCoreMLModelProfilingResult *)profilingInfoForOperationsAtPaths:(NSArray<ETCoreMLModelStructurePath *> *)paths
                                                                     options:(MLPredictionOptions *)options
                                                                      inputs:(id<MLFeatureProvider>)inputs
                                                                modelOutputs:(NSArray<MLMultiArray *> *_Nullable __autoreleasing *_Nonnull)modelOutputs
                                                                       error:(NSError* __autoreleasing *)error {
#if MODEL_PROFILING_IS_AVAILABLE
    uint64_t modelExecutionStartTime = mach_absolute_time();
    id<MLFeatureProvider> outputFeatures = [self.model predictionFromFeatures:inputs options:options error:error];
    uint64_t modelExecutionEndTime = mach_absolute_time();
    if (!modelOutputs) {
        return nil;
    }
    
    if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *)) {
        ETCoreMLModelProfilingResult *profilingInfos = get_profiling_infos_for_paths([NSSet setWithArray:paths],
                                                                                     self.topologicallySortedOperations,
                                                                                     self.operationToPathMap,
                                                                                     self.profilingInfos,
                                                                                     modelExecutionStartTime,
                                                                                     modelExecutionEndTime);
        
        
        
        if (outputFeatures) {
            set_model_outputs(outputFeatures, self.outputNames, modelOutputs);
        }
        
        return profilingInfos;
    }
#endif
    return nil;
}

- (nullable ETCoreMLModelProfilingResult *)profilingInfoForOperationsAtPaths:(MLPredictionOptions *)options
                                                                      inputs:(id<MLFeatureProvider>)inputs
                                                                modelOutputs:(NSArray<MLMultiArray *> *_Nullable __autoreleasing *_Nonnull)modelOutputs
                                                                       error:(NSError* __autoreleasing *)error {
#if MODEL_PROFILING_IS_AVAILABLE
    if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *)) {
        __block NSMutableArray<ETCoreMLModelStructurePath *> *paths = [NSMutableArray array];
        visit_program_operation(self.computePlan.modelStructure, ^BOOL(MLModelStructureProgramOperation *operation, ETCoreMLModelStructurePath *path) {
            if (!is_const_operation(operation)) {
                [paths addObject:path];
            }
            return YES;
        });

        return [self profilingInfoForOperationsAtPaths:paths
                                               options:options
                                                inputs:inputs
                                          modelOutputs:modelOutputs
                                                 error:error];
    }
#endif
    return nil;
}

@end
