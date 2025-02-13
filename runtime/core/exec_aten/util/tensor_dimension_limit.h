/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch::runtime {
/**
 * The expected output size may not be the existing size of any inputs and
 * outputs if the operator supports both broadcast and dynamic shape.
 * Therefore such operators needs extra space to store the calculated expected
 * output size. such dynamic allocation is troublesome in executorch so we can
 * just hard code a static value of a relatively small value because users
 * don't create high dimensional tensors.
 */
constexpr size_t kTensorDimensionLimit = 16;
} // namespace executorch::runtime
