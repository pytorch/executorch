/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

// Debug switch for operator registry
#if defined(ET_OP_REGISTRY_DEBUG)
#include <ostream>
#endif

#define ET_LOG_KERNEL_KEY(k)      \
  ET_LOG(                         \
      Info,                       \
      "key: %s, is_fallback: %s", \
      k.data(),                   \
      k.is_fallback() ? "true" : "false");
#define ET_LOG_TENSOR_META(meta_list)                                \
  for (const auto& meta : meta_list) {                               \
    ET_LOG(Info, "dtype: %d | dim order: [", int(meta.dtype_));      \
    for (size_t i = 0; i < meta.dim_order_.size(); i++) {            \
      ET_LOG(Info, "%d,", static_cast<int32_t>(meta.dim_order_[i])); \
    }                                                                \
    ET_LOG(Info, "]");                                               \
  }

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

class KernelRuntimeContext; // Forward declaration
using OpFunction = void (*)(KernelRuntimeContext&, Span<EValue*>);

/**
 * Dtype and dim order metadata for a Tensor argument to an operator.
 * Used by the Executor to hold the tensor metadata info and retrieve kernel.
 */
struct TensorMeta {
  executorch::aten::ScalarType dtype_;
  Span<executorch::aten::DimOrderType> dim_order_;

  TensorMeta() = default;
  TensorMeta(
      executorch::aten::ScalarType dtype,
      Span<executorch::aten::DimOrderType> order)
      : dtype_(dtype), dim_order_(order) {}

  bool operator==(const TensorMeta& other) const {
    return this->equals(other);
  }

  bool operator!=(const TensorMeta& other) const {
    return !this->equals(other);
  }

  bool equals(const TensorMeta& other) const {
    if (dtype_ != other.dtype_) {
      return false;
    }
    if (dim_order_.size() != other.dim_order_.size()) {
      return false;
    }
    for (size_t i = 0; i < dim_order_.size(); i++) {
      if (dim_order_[i] != other.dim_order_[i]) {
        return false;
      }
    }
    return true;
  }

#if defined(ET_OP_REGISTRY_DEBUG)
  friend std::ostream& operator<<(std::ostream& os, const TensorMeta& meta) {
    os << "dtype: " << int(meta.dtype_) << " | dim order: [";
    for (int i = 0; i < meta.dim_order_.size(); i++) {
      os << static_cast<int32_t>(meta.dim_order_[i]) << ", ";
    }
    os << "]";
    return os;
  }
#endif
};

/**
 * Describes which dtype & dim order specialized kernel to be bound to an
 * operator.
 *
 * Kernel key data is a string with the format:
 *
 *     "v<version>/<tensor_meta>|<tensor_meta>..."
 *
 * The version is v1 for now. If the kernel key format changes, update the
 * version to avoid breaking pre-existing kernel keys.
 *
 * Each tensor_meta has the following format: "<dtype>;<dim_order,...>"
 *
 * Example kernel key data: "v1/7;0,1,2,3|1;0,1,2,3,4,5,6,7"
 *
 * This has two tensors: the first with dtype=7 and dim order 0,1,2,3, and the
 * second with dtype=1 and dim order 0,1,2,3,4,5,6,7.
 *
 * IMPORTANT:
 * Users should not construct a kernel key manually. Instead, it should be
 * generated from kernel yaml.
 */
struct KernelKey {
 public:
  /**
   * Creates a fallback (non-specialized) kernel key: this kernel can be used
   * for all input tensor dtypes and dim orders if the specialized kernel is not
   * registered.
   */
  KernelKey() = default;

  /**
   * Creates a specialized (non-fallback) kernel key that matches a specific
   * set of input tensor dtypes and dim orders. See the class comment for the
   * expected format of `kernel_key_data`.
   */
  /* implicit */ KernelKey(const char* kernel_key_data)
      : kernel_key_data_(kernel_key_data) {}

  bool operator==(const KernelKey& other) const {
    return this->equals(other);
  }

  bool operator!=(const KernelKey& other) const {
    return !this->equals(other);
  }

  bool equals(const KernelKey& other) const {
    if (is_fallback() != other.is_fallback()) {
      return false;
    }
    if (is_fallback()) {
      return true;
    }
    return strcmp(kernel_key_data_, other.kernel_key_data_) == 0;
  }

  bool is_fallback() const {
    return kernel_key_data_ == nullptr;
  }

  const char* data() const {
    return kernel_key_data_;
  }

#if defined(ET_OP_REGISTRY_DEBUG)
  friend std::ostream& operator<<(std::ostream& os, const KernelKey& key) {
    os << key.kernel_key_data_ << std::endl;
    return os;
  }
#endif

 private:
  const char* kernel_key_data_ = nullptr;
};

/**
 * Struct that bundles a kernel key, a function and an op name together. An
 * `Operator` may have more than one `Kernel` (maximum kMaxNumOfKernelPerOp) and
 * they should have the same op name and different kernel key. A "fallback"
 * kernel may or may not live in an `Operator`.
 */
struct Kernel {
  const char* name_;
  // String representation of kernel key, with the same format as
  // KernelKey.to_string_representation()
  // Data is not owned by the Kernel struct.
  KernelKey kernel_key_;
  OpFunction op_;
  /**
   * We are doing a copy of the string pointer instead of duplicating the string
   * itself, we require the lifetime of the operator name to be at least as long
   * as the operator registry.
   */
  explicit Kernel(const char* name, OpFunction func) : name_(name), op_(func) {}

  explicit Kernel(const char* name, KernelKey key, OpFunction func)
      : name_(name), kernel_key_(key), op_(func) {}

  Kernel() {}
};

namespace internal {

/**
 * A make_kernel_key_string buffer size that is large enough to hold a kernel
 * key string with 16 tensors of 16 dimensions, plus the trailing NUL byte.
 */
constexpr size_t kKernelKeyBufSize = 659;

/**
 * Given the list of input tensor dtypes + dim orders, writes the kernel key
 * string into the buffer. Returns an error if the buffer is too small or if the
 * tensors cannot be represented as a valid key string.
 */
Error make_kernel_key_string(
    Span<const TensorMeta> key,
    char* buf,
    size_t buf_size);

} // namespace internal

/**
 * Checks whether an operator exists with a given name and TensorMeta list. When
 * TensorMeta is empty, it means this op does not have specialized kernels, so
 * it checks whether it has any fallback kernels.
 */
bool registry_has_op_function(
    const char* name,
    Span<const TensorMeta> meta_list = {});

/**
 * Returns the operator with a given name and TensorMeta list, if present.
 */
::executorch::runtime::Result<OpFunction> get_op_function_from_registry(
    const char* name,
    Span<const TensorMeta> meta_list = {});

/**
 * Returns all registered kernels.
 */
Span<const Kernel> get_registered_kernels();

/**
 * Registers the provided kernels.
 *
 * @param[in] kernels Kernel objects to register.
 * @retval Error::Ok always. Panics on error. This function needs to return a
 *     non-void type to run at static initialization time.
 */
ET_NODISCARD Error register_kernels(const Span<const Kernel>);

/**
 * Registers a single kernel.
 *
 * @param[in] kernel Kernel object to register.
 * @retval Error::Ok always. Panics on error. This function needs to return a
 *     non-void type to run at static initialization time.
 */
ET_NODISCARD inline Error register_kernel(const Kernel& kernel) {
  return register_kernels({&kernel, 1});
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::Kernel;
using ::executorch::ET_RUNTIME_NAMESPACE::KernelKey;
using ::executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using ::executorch::ET_RUNTIME_NAMESPACE::OpFunction;
using ::executorch::ET_RUNTIME_NAMESPACE::TensorMeta;
using KernelRuntimeContext =
    ::executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;

inline ::executorch::runtime::Error register_kernels(ArrayRef<Kernel> kernels) {
  return ::executorch::ET_RUNTIME_NAMESPACE::register_kernels(
      {kernels.data(), kernels.size()});
}
inline OpFunction getOpsFn(
    const char* name,
    ArrayRef<TensorMeta> meta_list = {}) {
  auto result =
      ::executorch::ET_RUNTIME_NAMESPACE::get_op_function_from_registry(
          name, {meta_list.data(), meta_list.size()});
  ET_CHECK(result.ok()); // get_op_function_from_registry() logs details.
  return *result;
}
inline bool hasOpsFn(const char* name, ArrayRef<TensorMeta> meta_list = {}) {
  return ::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(
      name, {meta_list.data(), meta_list.size()});
}
inline ArrayRef<Kernel> get_kernels() {
  Span<const Kernel> kernels =
      ::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels();
  return ArrayRef<Kernel>(kernels.data(), kernels.size());
}
} // namespace executor
} // namespace torch
