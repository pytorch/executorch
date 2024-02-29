/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
/// \file runtime/kernel/make_aten_functor_from_et_functor.h
/// Defines a template that can be used to create a ATen version of an unboxed
/// ExecuTorch kernel.
//===----------------------------------------------------------------------===//

#pragma once
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif
#include <torch/torch.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/meta_programming.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch {
namespace executor {

class KernelRuntimeContext; // Forward declaration
using RuntimeContext = KernelRuntimeContext; // TODO(T147221312): Remove

template <typename T>
struct type_map final {
    using type = T;
};

template <>
struct type_map<torch::executor::Tensor&> final {
    using type = at::Tensor&;
};

template <>
struct type_map<const torch::executor::Tensor&> final {
    using type = const at::Tensor&;
};

template <typename F, typename T>
struct type_convert final {
  public:
    F val;
    type_convert(F value) : val(value) {}
    T call() {
        return static_cast<T>(val);
    }
};

template <>
struct type_convert<at::Tensor&, torch::executor::Tensor&> final {
  public:
    at::Tensor& val;
    std::unique_ptr<ManagedTensor> managed_tensor;
    torch::executor::Tensor converted;
    std::vector<exec_aten::SizesType> sizes;
    type_convert(at::Tensor& value) : val(value), converted(torch::executor::Tensor(nullptr)){
        for (auto size : val.sizes()) {
          sizes.push_back(size);
        }
        torch::executor::ScalarType scalar_type = static_cast<torch::executor::ScalarType>(val.scalar_type());
        managed_tensor = std::make_unique<ManagedTensor>(val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
        converted = managed_tensor->get_aliasing_tensor();
    }
    torch::executor::Tensor& call() {
        return converted;
    }
};

template <>
struct type_convert<const at::Tensor&, const torch::executor::Tensor&> final {
  public:
    const at::Tensor& val;
    std::unique_ptr<ManagedTensor> managed_tensor;
    torch::executor::Tensor converted;
    std::vector<exec_aten::SizesType> sizes;
    type_convert(const at::Tensor& value) : val(value), converted(torch::executor::Tensor(nullptr)) {
        for (auto size : val.sizes()) {
          sizes.push_back(size);
        }
        torch::executor::ScalarType scalar_type = static_cast<torch::executor::ScalarType>(val.scalar_type());
        managed_tensor = std::make_unique<ManagedTensor>(val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
        converted = managed_tensor->get_aliasing_tensor();
    }
    const torch::executor::Tensor& call() {
        return converted;
    }
};

template <>
struct type_convert<torch::executor::Tensor&, at::Tensor&> final {
  public:
    torch::executor::Tensor& val;
    at::Tensor converted;
    std::vector<int64_t> sizes;
    type_convert(torch::executor::Tensor& value) : val(value) {
        for (auto size : val.sizes()) {
          sizes.push_back(size);
        }
        c10::ScalarType scalar_type = static_cast<c10::ScalarType>(val.scalar_type());
        converted = at::from_blob(val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
    }
    at::Tensor& call() {
        return converted;
    }
};

template <>
struct type_convert<const torch::executor::Tensor&, const at::Tensor&> final {
  public:
    const torch::executor::Tensor& val;
    at::Tensor converted;
    std::vector<int64_t> sizes;
    type_convert(const torch::executor::Tensor& value) : val(value) {
        for (auto size : val.sizes()) {
          sizes.push_back(size);
        }
        c10::ScalarType scalar_type = static_cast<c10::ScalarType>(val.scalar_type());
        converted = at::from_blob(val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
    }
    const at::Tensor& call() {
        return converted;
    }
};

template<class F, F f> struct wrapper_impl;

template<class R, class... Args, R(*f)(Args...)>
struct wrapper_impl<R(*)(Args...), f> {
    using ReturnType = typename type_map<R>::type;
    using TupleArgsType = std::tuple<type_convert<typename type_map<Args>::type, Args>...>;
    static constexpr size_t num_args = sizeof...(Args);

    static ReturnType wrap(typename type_map<Args>::type... args) {
        // stuff
        TupleArgsType converts = std::tuple(type_convert<typename type_map<Args>::type, Args>(args)...);
        R result = call_functor_with_args(converts, std::make_index_sequence<num_args>());
        return type_convert<R, ReturnType>(result).call();
    }

    private:
    template <size_t... indices>
    static R call_functor_with_args(TupleArgsType& converts, std::index_sequence<indices...>) {
        return f(std::get<indices>(converts).call()...);
    }
};
} // namespace executor
} // namespace torch
