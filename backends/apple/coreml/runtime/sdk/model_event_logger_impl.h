//
// model_event_logger_impl.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import "model_event_logger.h"
#import <CoreML/CoreML.h>

namespace executorch::runtime {
class EventTracer;
}

namespace executorchcoreml {
/// A class implementing the `ModelEventLogger` protocol.
///
/// It doesn't own the tracer object, the object must not be used if the tracer is gone.
class ModelEventLoggerImpl final : public ModelEventLogger {
public:
    /// Construct a `ModelEventLoggerImpl` from the `EventTracer`.
    explicit ModelEventLoggerImpl(::executorch::runtime::EventTracer* tracer) : tracer_(tracer) { }

    /// Logs profiling infos.
    ///
    /// @param op_path_to_profiling_info_map A dictionary with the operation path as the key and operation's profiling
    /// info as the value.
    /// @param op_path_to_debug_symbol_name_map A dictionary with the operation path as the key and the debug symbol
    /// name as the value.
    void log_profiling_infos(
        NSDictionary<ETCoreMLModelStructurePath*, ETCoreMLOperationProfilingInfo*>* op_path_to_profiling_info_map,
        NSDictionary<ETCoreMLModelStructurePath*, NSString*>* op_path_to_debug_symbol_name_map) const noexcept override;

    /// Logs intermediate tensor values.
    ///
    /// @param op_path_to_value_map A dictionary with the operation path as the key and the operation's value as the
    /// value.
    /// @param op_path_to_debug_symbol_name_map A dictionary with the operation path as the key and the debug symbol
    /// name as the value.
    void log_intermediate_tensors(
        NSDictionary<ETCoreMLModelStructurePath*, MLMultiArray*>* op_path_to_value_map,
        NSDictionary<ETCoreMLModelStructurePath*, NSString*>* op_path_to_debug_symbol_map) const noexcept override;

private:
    ::executorch::runtime::EventTracer* tracer_;
};
} // namespace executorchcoreml
