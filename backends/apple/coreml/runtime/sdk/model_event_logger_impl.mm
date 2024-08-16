//
// model_event_logger_impl.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "model_event_logger_impl.h"

#import "ETCoreMLModelStructurePath.h"
#import "ETCoreMLOperationProfilingInfo.h"
#import <executorch/runtime/core/event_tracer.h>
#import "objc_array_util.h"
#import <mach/mach_time.h>
#import <numeric>
#import "MLMultiArray_Copy.h"

namespace {

using namespace torch::executor;

uint64_t time_units_to_nano_seconds(uint64_t time_units) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSCAssert(mach_timebase_info(&info) == KERN_SUCCESS, @"ModelEventLogger: Failed to get time base.");
    });
    
    return time_units * info.numer / info.denom;
}

std::optional<ScalarType> to_scalar_type(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeFloat16: {
            return ScalarType::Half;
        }
        case MLMultiArrayDataTypeFloat32: {
            return ScalarType::Float;
        }
        case MLMultiArrayDataTypeDouble: {
            return ScalarType::Double;
        }
        case MLMultiArrayDataTypeInt32: {
            return ScalarType::Int;
        }
        default: {
            return std::nullopt;
        }
    }
}

MLMultiArrayDataType get_supported_data_type(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeFloat16: {
            return MLMultiArrayDataTypeFloat32;
        }
        default: {
            return data_type;
        }
    }
}

bool is_packed(NSArray<NSNumber *> *shape, NSArray<NSNumber *> *strides) {
    if (shape.count == 0) {
        return true;
    }
    size_t product = 1;
    for (size_t i = shape.count; i > 0; i--) {
        if (![strides[i - 1] isEqual:@(product)]) {
            return false;
        }
        product *= shape[i - 1].unsignedLongValue;
    }

    return true;
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
                                                    NSDictionary<ETCoreMLModelStructurePath*, NSString*> *op_path_to_debug_symbol_name_map) const noexcept {
    [op_path_to_value_map enumerateKeysAndObjectsUsingBlock:^(ETCoreMLModelStructurePath *path,
                                                              MLMultiArray *intermediate_value,
                                                              BOOL * _Nonnull __unused stop) {
        using namespace torch::executor;

        @autoreleasepool {
            NSString *debug_symbol = op_path_to_debug_symbol_name_map[path];
            if (debug_symbol == nil) {
                return;
            }

            MLMultiArray *value = op_path_to_value_map[path];
            if (value == nil || value.count == 0) {
                return;
            }

            MLMultiArray *supported_value = value;
            NSArray<NSNumber *> *shape = supported_value.shape; 
            NSError *local_error = nil;
            MLMultiArrayDataType data_type = get_supported_data_type(value.dataType);

            if (!is_packed(shape, value.strides) || (supported_value.dataType != data_type)) {
                supported_value = [[MLMultiArray alloc] initWithShape:shape
                                                             dataType:data_type
                                                                 error:&local_error];
                NSCAssert(supported_value != nil, 
                          @"ModelEventLoggerImpl: Failed to create packed multiarray with shape=%@, dataType=%ld, error=%@.",
                          shape,
                          static_cast<long>(value.dataType),
                          local_error);
                [value copyInto:supported_value];
            }


            [supported_value getBytesWithHandler:^(const void * _Nonnull bytes, NSInteger size) {
                auto sizes = to_vector<TensorImpl::SizesType>(shape);
                auto strides = to_vector<TensorImpl::StridesType>(supported_value.strides);
                auto scalar_type = to_scalar_type(data_type);
                auto dim_order = std::vector<TensorImpl::DimOrderType>(shape.count);
                std::iota(std::begin(dim_order), std::end(dim_order), 0);

                NSCAssert(scalar_type.has_value(), @"ModelEventLoggerImpl: MultiArray dataType=%ld is not supported.", static_cast<long>(data_type));
                auto tensor_impl = TensorImpl(
                    scalar_type.value(),
                    static_cast<ssize_t>(sizes.size()),
                    sizes.data(),
                    const_cast<void *>(bytes),
                    dim_order.data(),
                    strides.data());
                auto tensor = Tensor(&tensor_impl);
                tracer_->log_intermediate_output_delegate(debug_symbol.UTF8String, -1, tensor);
            }];
        }
    }];
}
} // namespace executorchcoreml
