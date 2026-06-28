/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * AdamW optimizer to perform on-device training. This is an adaptation of the
 * PyTorch AdamW implementation (Loshchilov & Hutter, 2019) that decouples
 * weight decay from the gradient-based update. Per-parameter state consists of
 * first and second moment running averages and a scalar step counter used for
 * bias correction.
 */
#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace extension {
namespace training {
namespace optimizer {

/**
 * AdamW optimizer state. Holds the two moment buffers and the step counter for
 * a single parameter, to be reused across optimizer steps.
 */
class ET_EXPERIMENTAL AdamWParamState {
 public:
  /**
   * Constructs a new AdamW param state.
   *
   * @param[in] exp_avg The first moment (EMA of gradients) buffer.
   * @param[in] exp_avg_sq The second moment (EMA of squared gradients) buffer.
   */
  AdamWParamState(
      executorch::extension::TensorPtr exp_avg,
      executorch::extension::TensorPtr exp_avg_sq)
      : exp_avg_(std::move(exp_avg)),
        exp_avg_sq_(std::move(exp_avg_sq)),
        step_count_(0) {}

  executorch::aten::Tensor& exp_avg() {
    return *exp_avg_;
  }

  executorch::aten::Tensor& exp_avg_sq() {
    return *exp_avg_sq_;
  }

  int64_t step_count() const {
    return step_count_;
  }

  void increment_step_count() {
    ++step_count_;
  }

 private:
  executorch::extension::TensorPtr exp_avg_;
  executorch::extension::TensorPtr exp_avg_sq_;
  int64_t step_count_;
};

/**
 * AdamW optimizer options. Hyperparameters for a given parameter group.
 */
class ET_EXPERIMENTAL AdamWOptions {
 public:
  /**
   * Constructs a new AdamW optimizer options.
   *
   * @param[in] lr The learning rate.
   * @param[in] beta1 Exponential decay rate for the first moment estimate.
   * @param[in] beta2 Exponential decay rate for the second moment estimate.
   * @param[in] eps Small constant added to the denominator for numerical
   *   stability.
   * @param[in] weight_decay Decoupled weight decay coefficient. Applied
   *   directly to the parameter (not folded into the gradient) per the AdamW
   *   formulation.
   */
  explicit AdamWOptions(
      double lr = 1e-3,
      double beta1 = 0.9,
      double beta2 = 0.999,
      double eps = 1e-8,
      double weight_decay = 1e-2)
      : lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        weight_decay_(weight_decay) {}

  std::unique_ptr<AdamWOptions> clone() const {
    return std::make_unique<AdamWOptions>(
        static_cast<const AdamWOptions&>(*this));
  }

  double lr() const {
    return lr_;
  }

  double beta1() const {
    return beta1_;
  }

  double beta2() const {
    return beta2_;
  }

  double eps() const {
    return eps_;
  }

  double weight_decay() const {
    return weight_decay_;
  }

 private:
  double lr_;
  double beta1_;
  double beta2_;
  double eps_;
  double weight_decay_;
};

/**
 * AdamW optimizer param group. Holds a set of named parameters and the options
 * governing their update.
 */
class ET_EXPERIMENTAL AdamWParamGroup {
 public:
  // NOTE: In order to store `AdamWParamGroup` in a `std::vector`, it has
  // to be copy-constructible.
  AdamWParamGroup(const AdamWParamGroup& param_group)
      : named_parameters_(param_group.named_parameters()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  AdamWParamGroup& operator=(const AdamWParamGroup& param_group) {
    this->named_parameters_ = param_group.named_parameters_;
    this->options_ =
        param_group.has_options() ? param_group.options().clone() : nullptr;
    return *this;
  }

  /* implicit */ AdamWParamGroup(
      const std::map<std::string_view, executorch::aten::Tensor>&
          named_parameters)
      : named_parameters_(named_parameters) {}
  AdamWParamGroup(
      const std::map<std::string_view, executorch::aten::Tensor>&
          named_parameters,
      std::unique_ptr<AdamWOptions> options)
      : named_parameters_(named_parameters), options_(std::move(options)) {}

  bool has_options() const;
  AdamWOptions& options();
  const AdamWOptions& options() const;
  void set_options(std::unique_ptr<AdamWOptions> options);
  const std::map<std::string_view, executorch::aten::Tensor>& named_parameters()
      const;

 private:
  std::map<std::string_view, executorch::aten::Tensor> named_parameters_;
  std::unique_ptr<AdamWOptions> options_;
};

/**
 * AdamW optimizer class. Performs the optimization step.
 */
class ET_EXPERIMENTAL AdamW {
 public:
  explicit AdamW(
      const std::vector<AdamWParamGroup>& param_groups,
      AdamWOptions defaults)
      : defaults_(defaults) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  explicit AdamW(
      const std::map<std::string_view, executorch::aten::Tensor>&
          named_parameters,
      AdamWOptions defaults)
      : AdamW({AdamWParamGroup(named_parameters)}, defaults) {}

  // Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const AdamWParamGroup& param_group);

  ~AdamW();

  /**
   * Performs the optimization step.
   *
   * @param[in] named_gradients The gradients of the tensors specified by the
   * fully qualified name.
   */
  ::executorch::runtime::Error step(
      const std::map<std::string_view, executorch::aten::Tensor>&
          named_gradients);

 private:
  std::vector<AdamWParamGroup> param_groups_;
  std::unordered_map<void*, std::unique_ptr<AdamWParamState>> state_;
  AdamWOptions defaults_;
};

} // namespace optimizer
} // namespace training
} // namespace extension
} // namespace executorch
