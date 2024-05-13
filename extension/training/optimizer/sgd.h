/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * SGD (stochastic gradient descent) optimizer to perform on-device training.
 * This uses the gradients calculated in the backwards pass of the loss function
 * and updates the parameters such that it minimizes the loss.
 *
 * This is similar to the Lite Interpreter implementation of the SGD optimizer,
 * but without the dependency on ATen Tensors and autograd.
 */
#pragma once

namespace torch {
namespace executor {
namespace optimizer {

/**
 * SGD optimizer state. This keeps track of the state of a given parameter to
 * be used in later epochs.
 */
class SGDParamState {};

/**
 * SGD optimizer options. This contains options for performing training on a
 * param group, such as the learning rate.
 */
class SGDOptions {};

/**
 * SGD optimizer param group. This contains the parameters and
 * the OptimizerOptions associated to it.
 */
class SGDParamGroup {};

/**
 * SGD optimizer class. This is responsible for performing the optimization
 * step.
 */
class SGD {};

} // namespace optimizer
} // namespace executor
} // namespace torch
