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

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <memory>

namespace torch {
namespace executor {
namespace optimizer {

using Tensor = exec_aten::Tensor;

/**
 * SGD optimizer state. This keeps track of the state of a given parameter to
 * be used in later epochs.
 */
class SGDParamState {
 public:
  /**
   * Constructs a new SGD param state.
   *
   * @param[in] momentum_buffer A tensor that stores the momentum at the last
   * epoch.
   */
  explicit SGDParamState(Tensor& momentum_buffer)
      : momentum_buffer_(momentum_buffer) {}

  Tensor& momentum_buffer() {
    return momentum_buffer_;
  }

 private:
  Tensor momentum_buffer_;
};

/**
 * SGD optimizer options. This contains options for performing training on a
 * param group, such as the learning rate.
 */
class SGDOptions {
 public:
  /**
   * Constructs a new SGD optimizer options.
   *
   * This is used for customizing the SGD optimizer for a given group of
   * parameters.
   *
   * @param[in] lr The learning rate. This is the factor applied to the gradient
   *   calculated from the loss function and used to update the parameters. A
   *   lower learning rate will result in a smaller step towards the minimum of
   * a loss function, and a higher learning rate will result in a larger step.
   * @param[in] momentum The momentum value. This is a used to accelerate the
   *   update step by using the gradients from previous epochs.
   * @param[in] dampening The dampening value. This is used in combination with
   *   momentum, and aims t o prevent the optimizer from taking steps that are
   *   too large when using the momentum.
   * @param[in] weight_decay The weight decay value. This is used as a
   *   regularization technique and is used to subtract a small fraction of the
   *   weight's value from itself at each step.
   * @param[in] nesterov Whether to use Nesterov momentum. If true, the
   *   optimizer uses the momentum of the current step and applies it to the
   *   training update. When false, the optimizer uses the momentum of the
   *   previous step and applies it to the training update.
   */
  explicit SGDOptions(
      double lr,
      double momentum = 0,
      double dampening = 0,
      double weight_decay = 0,
      bool nesterov = false)
      : lr_(lr),
        momentum_(momentum),
        dampening_(dampening),
        weight_decay_(weight_decay),
        nesterov_(nesterov) {}

  std::unique_ptr<SGDOptions> clone() const {
    return std::make_unique<SGDOptions>(static_cast<const SGDOptions&>(*this));
  }

  double lr() const {
    return lr_;
  }

  double momentum() const {
    return momentum_;
  }

  double dampening() const {
    return dampening_;
  }

  double weight_decay() const {
    return weight_decay_;
  }

  bool nesterov() const {
    return nesterov_;
  }

 private:
  double lr_;
  double momentum_;
  double dampening_;
  double weight_decay_;
  bool nesterov_;
};

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
