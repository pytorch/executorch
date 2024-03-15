//
// model_logging_options.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <string>

namespace executorchcoreml {
struct ModelLoggingOptions {
    /// If set to `true` then the delegate would log the profiling info of operations in the Program.
    bool log_profiling_info = false;
    /// If set to `true` then the delegate would log the value of operations the Program.
    bool log_intermediate_tensors = false;
};
}
