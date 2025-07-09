/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>

namespace executorch {
namespace backends {
namespace qnn {

/**
 * @brief Storing runtime option value.
 * @param is_set True when user calls set_option api to set option, else False.
 */
struct RuntimeOption {
  bool is_set;
  executorch::runtime::OptionValue value;
};

/**
 * @brief
 * Get the backend option.
 * This method checks both AOT option and runtime option.
 * If runtime option is provided, it will have a higher priority.
 *
 * @param aot_option The flatbuffer option under qc_compiler_spec.fbs.
 */

template <typename T>
T get_option(T aot_option);

} // namespace qnn
} // namespace backends
} // namespace executorch
