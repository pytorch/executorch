/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/optimizer/adamw.h>

#include <executorch/runtime/core/error.h>

#include <cmath>
#include <cstring>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using ::executorch::runtime::Error;

namespace executorch {
namespace extension {
namespace training {
namespace optimizer {

namespace {
// out[i] = a[i] + alpha * b[i]
void add_out_hack(
    const Tensor& a,
    const Tensor& b,
    const double alpha,
    Tensor& out) {
  auto a_ptr = a.const_data_ptr<float>();
  auto b_ptr = b.const_data_ptr<float>();
  auto out_ptr = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < a.numel(); ++i) {
    out_ptr[i] = a_ptr[i] + b_ptr[i] * alpha;
  }
}

// out[i] = a[i] * alpha
void mul_out_hack(const Tensor& a, const double alpha, Tensor& out) {
  auto a_ptr = a.const_data_ptr<float>();
  auto out_ptr = out.mutable_data_ptr<float>();
  for (size_t i = 0; i < a.numel(); ++i) {
    out_ptr[i] = a_ptr[i] * alpha;
  }
}

// Fused second-moment update: v[i] = beta2 * v[i] + (1 - beta2) * g[i]^2.
// Avoids materializing a separate g^2 tensor.
void addcmul_sq_out_hack(
    const Tensor& v,
    const Tensor& g,
    const double beta2,
    Tensor& out) {
  auto v_ptr = v.const_data_ptr<float>();
  auto g_ptr = g.const_data_ptr<float>();
  auto out_ptr = out.mutable_data_ptr<float>();
  const double one_minus_beta2 = 1.0 - beta2;
  for (size_t i = 0; i < v.numel(); ++i) {
    const double gi = static_cast<double>(g_ptr[i]);
    out_ptr[i] = static_cast<float>(
        static_cast<double>(v_ptr[i]) * beta2 + one_minus_beta2 * gi * gi);
  }
}

// Fused AdamW parameter update:
//   p[i] -= lr * (m[i] / bias_correction1) /
//           (sqrt(v[i] / bias_correction2) + eps)
// Performed in double precision internally to limit accumulated FP error on
// the division-by-sqrt path.
void adamw_update_hack(
    Tensor& p,
    const Tensor& m,
    const Tensor& v,
    const double lr,
    const double bias_correction1,
    const double bias_correction2,
    const double eps) {
  auto p_ptr = p.mutable_data_ptr<float>();
  auto m_ptr = m.const_data_ptr<float>();
  auto v_ptr = v.const_data_ptr<float>();
  const double inv_bc1 = 1.0 / bias_correction1;
  const double inv_sqrt_bc2 = 1.0 / std::sqrt(bias_correction2);
  for (size_t i = 0; i < p.numel(); ++i) {
    const double m_hat = static_cast<double>(m_ptr[i]) * inv_bc1;
    const double v_hat_sqrt =
        std::sqrt(static_cast<double>(v_ptr[i])) * inv_sqrt_bc2;
    p_ptr[i] = static_cast<float>(
        static_cast<double>(p_ptr[i]) - lr * m_hat / (v_hat_sqrt + eps));
  }
}
} // namespace

bool AdamWParamGroup::has_options() const {
  return options_ != nullptr;
}

AdamWOptions& AdamWParamGroup::options() {
  return *options_.get();
}

const AdamWOptions& AdamWParamGroup::options() const {
  return *options_.get();
}

void AdamWParamGroup::set_options(std::unique_ptr<AdamWOptions> options) {
  options_ = std::move(options);
}

const std::map<std::string_view, executorch::aten::Tensor>&
AdamWParamGroup::named_parameters() const {
  return named_parameters_;
}

void AdamW::add_param_group(const AdamWParamGroup& param_group) {
  AdamWParamGroup param_group_(param_group.named_parameters());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  param_groups_.emplace_back(std::move(param_group_));
}

Error AdamW::step(
    const std::map<std::string_view, executorch::aten::Tensor>&
        named_gradients) {
  for (auto& group : param_groups_) {
    auto& options = static_cast<AdamWOptions&>(group.options());
    const double lr = options.lr();
    const double beta1 = options.beta1();
    const double beta2 = options.beta2();
    const double eps = options.eps();
    const double weight_decay = options.weight_decay();

    for (auto param_iter = group.named_parameters().begin();
         param_iter != group.named_parameters().end();
         ++param_iter) {
      const auto& named_gradient = named_gradients.find(param_iter->first);
      if (named_gradient == named_gradients.end()) {
        continue;
      }
      auto g = named_gradient->second;
      auto p = param_iter->second;

      // Decoupled weight decay: p <- p - lr * weight_decay * p. Applied to
      // the parameter directly, BEFORE the moment-based update, and NOT
      // folded into the gradient. This is the defining property of AdamW
      // (Loshchilov & Hutter, 2019).
      if (weight_decay != 0.0) {
        add_out_hack(p, p, -lr * weight_decay, p);
      }

      // Look up or lazily allocate the per-parameter state (two moment
      // buffers sized and shaped like the gradient, plus a step counter).
      auto param_state_it = state_.find(p.unsafeGetTensorImpl());
      AdamWParamState* state_ptr = nullptr;
      if (param_state_it == state_.end()) {
        void* m_buf_ptr = malloc(g.nbytes());
        void* v_buf_ptr = malloc(g.nbytes());
        std::memset(m_buf_ptr, 0, g.nbytes());
        std::memset(v_buf_ptr, 0, g.nbytes());

        Tensor m_buf(nullptr);
        Tensor v_buf(nullptr);
#ifdef USE_ATEN_LIB
        std::vector<int64_t> sizes(g.sizes().begin(), g.sizes().end());
        m_buf = torch::from_blob(m_buf_ptr, sizes, g.scalar_type());
        v_buf = torch::from_blob(v_buf_ptr, sizes, g.scalar_type());
#else
        TensorImpl* m_impl = new TensorImpl(
            g.scalar_type(),
            g.sizes().size(),
            const_cast<TensorImpl::SizesType*>(g.sizes().data()),
            m_buf_ptr,
            const_cast<TensorImpl::DimOrderType*>(g.dim_order().data()));
        TensorImpl* v_impl = new TensorImpl(
            g.scalar_type(),
            g.sizes().size(),
            const_cast<TensorImpl::SizesType*>(g.sizes().data()),
            v_buf_ptr,
            const_cast<TensorImpl::DimOrderType*>(g.dim_order().data()));
        m_buf = Tensor(m_impl);
        v_buf = Tensor(v_impl);
#endif
        auto state = std::make_unique<AdamWParamState>(m_buf, v_buf);
        state_ptr = state.get();
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      } else {
        state_ptr = param_state_it->second.get();
      }

      state_ptr->increment_step_count();
      const int64_t step = state_ptr->step_count();

      Tensor& exp_avg = state_ptr->exp_avg();
      Tensor& exp_avg_sq = state_ptr->exp_avg_sq();

      // First moment: m <- beta1 * m + (1 - beta1) * g
      mul_out_hack(exp_avg, beta1, exp_avg);
      add_out_hack(exp_avg, g, 1.0 - beta1, exp_avg);

      // Second moment: v <- beta2 * v + (1 - beta2) * g^2
      addcmul_sq_out_hack(exp_avg_sq, g, beta2, exp_avg_sq);

      // Bias-corrected update.
      const double bias_correction1 = 1.0 - std::pow(beta1, step);
      const double bias_correction2 = 1.0 - std::pow(beta2, step);
      adamw_update_hack(
          p, exp_avg, exp_avg_sq, lr, bias_correction1, bias_correction2, eps);
    }
  }
  return Error::Ok;
}

AdamW::~AdamW() {
  for (const auto& state_kv : state_) {
    auto& state = *state_kv.second;
    free(state.exp_avg().unsafeGetTensorImpl()->mutable_data());
    free(state.exp_avg_sq().unsafeGetTensorImpl()->mutable_data());
#ifndef USE_ATEN_LIB
    delete state.exp_avg().unsafeGetTensorImpl();
    delete state.exp_avg_sq().unsafeGetTensorImpl();
#endif
  }
}

} // namespace optimizer
} // namespace training
} // namespace extension
} // namespace executorch
