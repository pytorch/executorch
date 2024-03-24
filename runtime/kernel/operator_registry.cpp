/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/operator_registry.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/system.h>
#include <cinttypes>

#include <executorch/runtime/platform/assert.h>

#include <executorch/kernels/portable/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/profiler.h>

namespace torch {
namespace executor {

OperatorRegistry& getOperatorRegistry();
OperatorRegistry& getOperatorRegistry() {
  static OperatorRegistry operator_registry;
  return operator_registry;
}

Error register_kernels(const ArrayRef<Kernel>& kernels) {
  Error success = getOperatorRegistry().register_kernels(kernels);
  if (success == Error::InvalidArgument || success == Error::Internal) {
    ET_CHECK_MSG(
        false,
        "Kernel registration failed with error %" PRIu32
        ", see error log for details.",
        static_cast<uint32_t>(success));
  }
  return success;
}

Error OperatorRegistry::register_kernels(const ArrayRef<Kernel>& kernels) {
  // Operator registration happens in static initialization time when PAL init
  // may or may not happen already. Here we are assuming et_pal_init() doesn't
  // have any side effect even if falled multiple times.
  ::et_pal_init();

  if (kernels.size() + this->num_kernels_ > kMaxNumOfKernels) {
    ET_LOG(
        Error,
        "The total number of kernels to be registered is larger than the limit %" PRIu32
        ". %" PRIu32
        " kernels are already registered and we're trying to register another %" PRIu32
        " kernels.",
        kMaxNumOfKernels,
        (uint32_t)this->num_kernels_,
        (uint32_t)kernels.size());
    ET_LOG(Error, "======== Kernels already in the registry: ========");
    for (size_t i = 0; i < this->num_kernels_; i++) {
      ET_LOG(Error, "%s", this->kernels_[i].name_);
      ET_LOG_KERNEL_KEY(this->kernels_[i].kernel_key_);
    }
    ET_LOG(Error, "======== Kernels being registered: ========");
    for (size_t i = 0; i < kernels.size(); i++) {
      ET_LOG(Error, "%s", kernels[i].name_);
      ET_LOG_KERNEL_KEY(kernels[i].kernel_key_);
    }
    return Error::Internal;
  }
  // for debugging purpose
  const char* lib_name = et_pal_get_shared_library_name(kernels.data());

  for (const auto& kernel : kernels) {
    // linear search. This is fine if the number of kernels are small.
    for (int32_t i = 0; i < this->num_kernels_; i++) {
      Kernel k = this->kernels_[i];
      if (strcmp(kernel.name_, k.name_) == 0 &&
          kernel.kernel_key_ == k.kernel_key_) {
        ET_LOG(Error, "Re-registering %s, from %s", k.name_, lib_name);
        ET_LOG_KERNEL_KEY(k.kernel_key_);
        return Error::InvalidArgument;
      }
    }
    this->kernels_[this->num_kernels_++] = kernel;
  }
  ET_LOG(
      Debug,
      "Successfully registered all kernels from shared library: %s",
      lib_name);

  return Error::Ok;
}

bool hasOpsFn(const char* name, ArrayRef<TensorMeta> kernel_key) {
  return getOperatorRegistry().hasOpsFn(name, kernel_key);
}

static int copy_char_as_number_to_buf(char num, char* buf) {
  if ((char)num < 10) {
    *buf = '0' + (char)num;
    buf += 1;
    return 1;
  } else {
    *buf = '0' + ((char)num) / 10;
    buf += 1;
    *buf = '0' + ((char)num) % 10;
    buf += 1;
    return 2;
  }
}

void make_kernel_key_string(ArrayRef<TensorMeta> key, char* buf);

void make_kernel_key_string(ArrayRef<TensorMeta> key, char* buf) {
  if (key.empty()) {
    // If no tensor is present in an op, kernel key does not apply
    return;
  }
  strncpy(buf, "v1/", 3);
  buf += 3;
  for (size_t i = 0; i < key.size(); i++) {
    auto& meta = key[i];
    buf += copy_char_as_number_to_buf((char)meta.dtype_, buf);
    *buf = ';';
    buf += 1;
    for (int j = 0; j < meta.dim_order_.size(); j++) {
      buf += copy_char_as_number_to_buf((char)meta.dim_order_[j], buf);
      if (j != meta.dim_order_.size() - 1) {
        *buf = ',';
        buf += 1;
      }
    }
    *buf = (i < (key.size() - 1)) ? '|' : 0x00;
    buf += 1;
  }
}

// This function is used to register edge dialect kernels on demand.
// edge dialect ops required by every ET model in edge dialect. To avoid
// duplicated registration and ET biniary size regression, we lazy register them
// on demand.
bool register_edge_dialect_ops_on_demand(const char* name);

bool register_edge_dialect_ops_on_demand(const char* name) {
  if (strcmp(name, "dim_order_ops::_to_dim_order_copy.out") == 0) {
    using KernelArrayRef =
        ::torch::executor::ArrayRef<::torch::executor::Kernel>;

    static Kernel kernels_to_register[] = {

        Kernel(
            "dim_order_ops::_to_dim_order_copy.out",
            [](torch::executor::KernelRuntimeContext& context, EValue** stack) {
              EValue& self = *stack[0];
              EValue& non_blocking = *stack[1];
              EValue& dim_order = *stack[2];
              EValue& out = *stack[3];
              const exec_aten::Tensor& self_base = self.to<exec_aten::Tensor>();
              bool non_blocking_base = non_blocking.to<bool>();

              exec_aten::optional<exec_aten::ArrayRef<int64_t>>
                  dim_order_opt_out =
                      dim_order.toOptional<exec_aten::ArrayRef<int64_t>>();

              exec_aten::Tensor& out_base = out.to<exec_aten::Tensor>();

#ifdef USE_ATEN_LIB
              torch::executor::native::_to_dim_order_copy_out(
                  self_base, non_blocking_base, dim_order_opt_out, out_base);
#else
              torch::executor::native::_to_dim_order_copy_out(
                  context,
                  self_base,
                  non_blocking_base,
                  dim_order_opt_out,
                  out_base);
#endif
            }), // Generated kernels
    };

    // Explicitly convert to ArrayRef, so that the API can take an empty C array
    // of Kernels.
    static KernelArrayRef kernel_array_ref(
        kernels_to_register,
        kernels_to_register + sizeof(kernels_to_register) / sizeof(Kernel));

    // Return value not used. Keep the static variable assignment to register
    // kernels in static initialization time.
    static auto success_with_kernel_reg = register_kernels(kernel_array_ref);
    if (success_with_kernel_reg == Error::Ok) {
      return true;
    }
  }
  return false;
}

bool OperatorRegistry::hasOpsFn(
    const char* name,
    ArrayRef<TensorMeta> meta_list) {
  char buf[KernelKey::MAX_SIZE] = {0};
  make_kernel_key_string(meta_list, buf);
  KernelKey kernel_key = KernelKey(buf);

  for (size_t idx = 0; idx < this->num_kernels_; idx++) {
    if (strcmp(this->kernels_[idx].name_, name) == 0) {
      if (this->kernels_[idx].kernel_key_.is_fallback() ||
          this->kernels_[idx].kernel_key_ == kernel_key) {
        return true;
      }
    }
  }

  return register_edge_dialect_ops_on_demand(name);
}

const OpFunction& getOpsFn(const char* name, ArrayRef<TensorMeta> kernel_key) {
  return getOperatorRegistry().getOpsFn(name, kernel_key);
}

const OpFunction& OperatorRegistry::getOpsFn(
    const char* name,
    ArrayRef<TensorMeta> meta_list) {
  char buf[KernelKey::MAX_SIZE] = {0};
  make_kernel_key_string(meta_list, buf);
  KernelKey kernel_key = KernelKey(buf);

  int32_t fallback_idx = -1;
  for (size_t idx = 0; idx < this->num_kernels_; idx++) {
    if (strcmp(this->kernels_[idx].name_, name) == 0) {
      if (this->kernels_[idx].kernel_key_ == kernel_key) {
        return this->kernels_[idx].op_;
      }
      if (this->kernels_[idx].kernel_key_.is_fallback()) {
        fallback_idx = idx;
      }
    }
  }

  if (fallback_idx != -1) {
    return this->kernels_[fallback_idx].op_;
  }

  // If no kernel is found, check whether if it is a edge dialect operator, and
  // register the kernel on demand.
  bool registered = register_edge_dialect_ops_on_demand(name);
  if (registered) {
    return getOpsFn(name, meta_list);
  }
  ET_CHECK_MSG(false, "kernel '%s' not found.", name);
  ET_LOG_TENSOR_META(meta_list);
}

ArrayRef<Kernel> get_kernels() {
  return getOperatorRegistry().get_kernels();
}

ArrayRef<Kernel> OperatorRegistry::get_kernels() {
  return ArrayRef<Kernel>(this->kernels_, this->num_kernels_);
}

} // namespace executor
} // namespace torch
