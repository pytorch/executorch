//
// model_event_logger_impl.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLModelStructurePath.h>
#import <ETCoreMLOperationProfilingInfo.h>
#import <executorch/runtime/core/event_tracer.h>
#import <mach/mach_time.h>
#import <model_event_logger_impl.h>

namespace {
uint64_t time_units_to_nano_seconds(uint64_t time_units) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSCAssert(mach_timebase_info(&info) == KERN_SUCCESS, @"ModelEventLogger: Failed to get time base.");
    });
    
    return time_units * info.numer / info.denom;
}

}

namespace executorchcoreml {

void ModelEventLoggerImpl::log_profiling_infos(NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *op_path_to_profiling_info_map,
                                               NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map) const noexcept {
    [op_path_to_profiling_info_map enumerateKeysAndObjectsUsingBlock:^(ETCoreMLModelStructurePath *path,
                                                                       ETCoreMLOperationProfilingInfo *profiling_info,
                                                                       BOOL * _Nonnull __unused stop) {
        if (profiling_info.estimatedExecutionEndTime == profiling_info.estimatedExecutionStartTime) {
            return;
        }
        uint64_t estimated_execution_start_time_in_ns = time_units_to_nano_seconds(profiling_info.estimatedExecutionStartTime);
        uint64_t estimated_execution_end_time_in_ns = time_units_to_nano_seconds(profiling_info.estimatedExecutionEndTime);
        NSString *symbol_name = op_path_to_debug_symbol_name_map[path];
        if (symbol_name == nil) {
            // We will use the operation output name as a placeholder for now.
            symbol_name = [NSString stringWithFormat:@"%@:%@", profiling_info.outputNames.firstObject, profiling_info.operatorName];
        }
        NSData *metadata = profiling_info.metadata;
        tracer_->log_profiling_delegate(symbol_name.UTF8String,
                                        -1,
                                        estimated_execution_start_time_in_ns,
                                        estimated_execution_end_time_in_ns,
                                        metadata.bytes,
                                        metadata.length);
        
    }];
}

void ModelEventLoggerImpl::log_intermediate_tensors(NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *op_path_to_value_map,
                                                    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map) const noexcept {
    //TODO: Implement logging for intermediate tensors once ExecuTorch has support.
}
} // namespace executorchcoreml
