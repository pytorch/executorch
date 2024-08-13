/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace torch {
namespace executor {
namespace training {
namespace optimizer {

bool SGDParamGroup::has_options() const {
  return options_ != nullptr;
}

SGDOptions& SGDParamGroup::options() {
  return *options_.get();
}

const SGDOptions& SGDParamGroup::options() const {
  return *options_.get();
}

void SGDParamGroup::set_options(std::unique_ptr<SGDOptions> options) {
  options_ = std::move(options);
}

Span<const char*> SGDParamGroup::param_names() {
  return param_names_;
}

const Span<const char*> SGDParamGroup::param_names() const {
  return param_names_;
}

Span<Tensor> SGDParamGroup::param_data() {
  return param_data_;
}

const Span<Tensor> SGDParamGroup::param_data() const {
  return param_data_;
}

void SGD::add_param_group(const SGDParamGroup& param_group) {
  SGDParamGroup param_group_(
      param_group.param_names(), param_group.param_data());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  param_groups_.emplace_back(std::move(param_group_));
}

Error SGD::step(Span<const char*> gradient_names, Span<Tensor> gradient_data) {
  // check that the number of gradient names matches the number of gradients
  ET_CHECK_OR_RETURN_ERROR(
      gradient_names.size() == gradient_data.size(),
      InvalidState,
      "Gradient names and gradients must have the same length.");

  RuntimeContext context;
  for (auto& group : param_groups_) {
    auto& options = static_cast<SGDOptions&>(group.options());
    auto weight_decay = options.weight_decay();
    auto momentum = options.momentum();
    auto dampening = options.dampening();
    auto nesterov = options.nesterov();

    for (int i = 0; i < group.param_names().size(); i++) {
      for (int j = 0; j < gradient_names.size(); j++) {
        // if param name and gradient name match, run the optimizer step
        if (strcmp(group.param_names()[i], gradient_names[j]) == 0) {
          auto d_p = gradient_data[j];
          auto p = group.param_data()[i];
          if (weight_decay != 0) {
            // uses weight_decay specified and adds it to the gradient
            torch::executor::aten::add_outf(context, d_p, p, weight_decay, d_p);
            if (context.failure_state() != Error::Ok) {
              return context.failure_state();
            }
          }
          if (momentum != 0) {
            Tensor buf(nullptr);
            auto param_state = state_.find(p.unsafeGetTensorImpl());
            // look for the momentum buffer for the given parameter. this is the
            // momentum as of the previous epoch
            if (param_state == state_.end()) {
              // create a new momentum buffer if it doesn't exist. this memory
              // needs to be freed when the optimizer is destroyed
              void* buf_ptr = malloc(d_p.nbytes());

#ifdef USE_ATEN_LIB
              std::vector<int64_t> sizes(
                  d_p.sizes().begin(), d_p.sizes().end());
              buf = torch::from_blob(buf_ptr, sizes, d_p.scalar_type());
#else
              TensorImpl* buf_impl = new TensorImpl(
                  d_p.scalar_type(),
                  d_p.sizes().size(),
                  const_cast<TensorImpl::SizesType*>(d_p.sizes().data()),
                  buf_ptr,
                  const_cast<TensorImpl::DimOrderType*>(
                      d_p.dim_order().data()));
              buf = Tensor(buf_impl);
#endif
              torch::executor::aten::clone_outf(
                  context, d_p, exec_aten::MemoryFormat::Contiguous, buf);
              if (context.failure_state() != Error::Ok) {
                return context.failure_state();
              }

              // save the state of the momentum buffer to be reused in later
              // epochs
              auto state = std::make_unique<SGDParamState>(buf);
              state_[p.unsafeGetTensorImpl()] = std::move(state);
            } else {
              buf = static_cast<SGDParamState&>(*param_state->second)
                        .momentum_buffer();

              // update the momentum buffer and apply dampening
              torch::executor::aten::mul_outf(context, buf, momentum, buf);
              if (context.failure_state() != Error::Ok) {
                return context.failure_state();
              }
              torch::executor::aten::add_outf(
                  context, buf, d_p, 1 - dampening, buf);
              if (context.failure_state() != Error::Ok) {
                return context.failure_state();
              }
            }
            if (nesterov) {
              // apply nesterov momentum
              torch::executor::aten::add_outf(context, d_p, buf, momentum, d_p);
              if (context.failure_state() != Error::Ok) {
                return context.failure_state();
              }
            } else {
              d_p = buf;
            }
          }
          // update the parameter using the gradient and learning rate
          torch::executor::aten::add_outf(
              context, p, d_p, -1 * options.lr(), p);
          if (context.failure_state() != Error::Ok) {
            return context.failure_state();
          }
          break;
        }
      }
    }
  }
  return Error::Ok;
}

SGD::~SGD() {
  for (const auto& state_kv : state_) {
    auto state_tensor = static_cast<SGDParamState&>(*state_kv.second);
    free(state_tensor.momentum_buffer().unsafeGetTensorImpl()->mutable_data());
#ifndef USE_ATEN_LIB
    delete state_tensor.momentum_buffer().unsafeGetTensorImpl();
#endif
  }
}
} // namespace optimizer
} // namespace training
} // namespace executor
} // namespace torch
