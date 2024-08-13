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

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace torch {
namespace executor {
namespace training {
namespace optimizer {

using Tensor = exec_aten::Tensor;
using TensorImpl = exec_aten::TensorImpl;
using ScalarType = exec_aten::ScalarType;

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
 * the SGDOptions associated to it.
 */
class SGDParamGroup {
 public:
  // NOTE: In order to store `SGDParamGroup` in a `std::vector`, it has
  // to be copy-constructible.
  SGDParamGroup(const SGDParamGroup& param_group)
      : param_data_(param_group.param_data()),
        param_names_(param_group.param_names()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  SGDParamGroup& operator=(const SGDParamGroup& param_group) {
    this->param_data_ = param_group.param_data();
    this->param_names_ = param_group.param_names();
    this->options_ =
        param_group.has_options() ? param_group.options().clone() : nullptr;
    return *this;
  }

  /**
   * This constructs a SGD param group. We expect that the two spans are of the
   * same size, and that for a given param data, its index in param_data is the
   * same as its param name in param_name.
   *
   * @param[in] param_names The names of the params for this group.
   * @param[in] param_data The tensors representing the param data.
   */
  /* implicit */ SGDParamGroup(
      Span<const char*> param_names,
      Span<Tensor> param_data)
      : param_data_(std::move(param_data)),
        param_names_(std::move(param_names)) {}
  SGDParamGroup(
      Span<const char*> param_names,
      Span<Tensor> param_data,
      std::unique_ptr<SGDOptions> options)
      : param_data_(std::move(param_data)),
        param_names_(std::move(param_names)),
        options_(std::move(options)) {}

  bool has_options() const;
  SGDOptions& options();
  const SGDOptions& options() const;
  void set_options(std::unique_ptr<SGDOptions> options);
  Span<const char*> param_names();
  const Span<const char*> param_names() const;
  Span<Tensor> param_data();
  const Span<Tensor> param_data() const;

 private:
  Span<Tensor> param_data_;
  Span<const char*> param_names_;
  std::unique_ptr<SGDOptions> options_;
};

/**
 * SGD optimizer class. This is responsible for performing the optimization
 * step.
 */
class SGD {
 public:
  explicit SGD(
      const std::vector<SGDParamGroup>& param_groups,
      SGDOptions defaults)
      : defaults_(std::make_unique<SGDOptions>(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  explicit SGD(
      Span<const char*> param_names,
      Span<Tensor> param_data,
      SGDOptions defaults)
      : SGD({SGDParamGroup(std::move(param_names), std::move(param_data))},
            defaults) {}

  // Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const SGDParamGroup& param_group);

  ~SGD();

  /**
   * Performs the optimization step.
   *
   * The two spans must be of the same size. It is expected that the gradient in
   * 'gradient_data' at index 'i' represents the gradient calculated in the loss
   * function for the parameter with the name in 'gradient_names' at index 'i'.
   *
   * @param[in] gradient_names The names of the params that matches the gradient
   *   in 'gradient_data' at the same index.
   * @param[in] gradient_data The gradient tensors to be used for optimization
   *   step.
   */
  Error step(Span<const char*> gradient_names, Span<Tensor> gradient_data);

 private:
  std::vector<SGDParamGroup> param_groups_;
  std::unordered_map<void*, std::unique_ptr<SGDParamState>> state_;
  std::unique_ptr<SGDOptions> defaults_;
};

} // namespace optimizer
} // namespace training
} // namespace executor
} // namespace torch
