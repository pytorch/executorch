/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/backend/options.h>

namespace executorch {
namespace backends {
namespace coreml {

/**
 * Builder for CoreML backend load-time options.
 *
 * This class provides a type-safe way to configure CoreML backend options
 * that are applied when loading a model. Use with LoadBackendOptionsMap
 * to pass options to Module::load().
 *
 * Example usage:
 * @code
 *   using executorch::backends::coreml::LoadOptionsBuilder;
 *
 *   LoadOptionsBuilder coreml_opts;
 *   coreml_opts.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_GPU);
 *
 *   LoadBackendOptionsMap map;
 *   map.set_options(coreml_opts);
 *
 *   module.load(method_name, map);
 * @endcode
 */
class LoadOptionsBuilder {
public:
    /**
     * Compute unit options for CoreML backend.
     *
     * String values are validated in the Objective-C++ backend delegate
     * using ETCoreMLStrings as the single source of truth.
     */
    enum class ComputeUnit {
        CPU_ONLY, // "cpu_only" - Run on CPU only
        CPU_AND_GPU, // "cpu_and_gpu" - Run on CPU and GPU
        CPU_AND_NE, // "cpu_and_ne" - Run on CPU and Neural Engine
        ALL // "all" - Run on all available compute units
    };

    /**
     * Sets the target compute unit for model execution.
     *
     * @param unit The compute unit to use.
     * @return Reference to this builder for chaining.
     */
    LoadOptionsBuilder& setComputeUnit(ComputeUnit unit) {
        const char* value = nullptr;
        switch (unit) {
            case ComputeUnit::CPU_ONLY:
                value = "cpu_only";
                break;
            case ComputeUnit::CPU_AND_GPU:
                value = "cpu_and_gpu";
                break;
            case ComputeUnit::CPU_AND_NE:
                value = "cpu_and_ne";
                break;
            case ComputeUnit::ALL:
                value = "all";
                break;
        }
        options_.set_option("compute_unit", value);
        return *this;
    }

    /**
     * Returns the backend identifier for this options builder.
     */
    static constexpr const char* backend_id() { return "CoreMLBackend"; }

    /**
     * Returns a view of the configured options.
     */
    ::executorch::runtime::Span<::executorch::runtime::BackendOption> view() { return options_.view(); }

private:
    ::executorch::runtime::BackendOptions<8> options_;
};

} // namespace coreml
} // namespace backends
} // namespace executorch
