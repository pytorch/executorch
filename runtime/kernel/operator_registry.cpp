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

namespace executorch {
namespace runtime {

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

  return false;
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
  ET_CHECK_MSG(false, "kernel '%s' not found.", name);
  ET_LOG_TENSOR_META(meta_list);
}

ArrayRef<Kernel> get_kernels() {
  return getOperatorRegistry().get_kernels();
}

ArrayRef<Kernel> OperatorRegistry::get_kernels() {
  return ArrayRef<Kernel>(this->kernels_, this->num_kernels_);
}

} // namespace runtime
} // namespace executorch
