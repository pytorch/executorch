//
// model_event_logger.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <CoreML/CoreML.h>

@class ETCoreMLModelStructurePath;
@class ETCoreMLOperationProfilingInfo;

namespace executorchcoreml {

/// An abstract class for logging model events (profiling/debugging) .
class ModelEventLogger {
public:
    inline ModelEventLogger() noexcept { }
    virtual ~ModelEventLogger() noexcept = default;

    /// Must log profiling infos.
    ///
    /// @param op_path_to_profiling_info_map A dictionary with the operation path as the key and operation's profiling
    /// info as the value.
    /// @param op_path_to_debug_symbol_name_map A dictionary with the operation path as the key and the symbol name as
    /// the value. The symbol name is the delegate handle.
    virtual void log_profiling_infos(
        NSDictionary<ETCoreMLModelStructurePath*, ETCoreMLOperationProfilingInfo*>* op_path_to_profiling_info_map,
        NSDictionary<ETCoreMLModelStructurePath*, NSString*>* op_path_to_debug_symbol_name_map) const noexcept = 0;

    /// Must log intermediate tensors.
    ///
    /// @param op_path_to_value_map A dictionary with the operation path as the key and the operation's value as the
    /// value.
    /// @param op_path_to_debug_symbol_name_map A dictionary with the operation path as the key and the debug symbol
    /// name as the value.
    virtual void log_intermediate_tensors(
        NSDictionary<ETCoreMLModelStructurePath*, MLMultiArray*>* op_path_to_value_map,
        NSDictionary<ETCoreMLModelStructurePath*, NSString*>* op_path_to_debug_symbol_name_map) const noexcept = 0;
};
}
