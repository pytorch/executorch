#include <executorch/core/OperatorRegistry.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/system.h>
#include <cinttypes>

#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

OperatorRegistry& getOperatorRegistry();
OperatorRegistry& getOperatorRegistry() {
  static OperatorRegistry operator_registry;
  return operator_registry;
}

Error register_operators(const ArrayRef<Operator>& operators) {
  Error success_with_op_reg =
      getOperatorRegistry().register_operators(operators);
  if (success_with_op_reg == Error::InvalidArgument ||
      success_with_op_reg == Error::Internal) {
    ET_CHECK_MSG(
        false,
        "Operator registration failed with error %" PRIu32
        ", see error log for details.",
        success_with_op_reg);
  }
  return success_with_op_reg;
}

Error OperatorRegistry::register_operators(
    const ArrayRef<Operator>& operators) {
  // Operator registration happens in static initialization time when PAL init
  // may or may not happen already. Here we are assuming et_pal_init() doesn't
  // have any side effect even if falled multiple times.
  ::et_pal_init();
  // Error out if number of operators exceeds the limit. Print all op name for
  // debugging
  if (this->operatorRegSize_ + operators.size() >= kOperatorTableMaxSize) {
    ET_LOG(Error, "======== Operators already in the registry: ========");
    for (size_t i = 0; i < this->operatorRegSize_; i++) {
      ET_LOG(Error, "%s", this->operators_table_[i].name_);
    }
    ET_LOG(Error, "======== Operators being registered: ========");
    for (size_t i = 0; i < operators.size(); i++) {
      ET_LOG(Error, "%s", operators[i].name_);
    }
    ET_LOG(
        Error,
        "The total number of operators to be registered is larger than the limit %" PRIu32
        ". %" PRIu32
        " operators are already registered and we're trying to register another %" PRIu32
        " operators.",
        kOperatorTableMaxSize,
        (uint32_t)this->operatorRegSize_,
        (uint32_t)operators.size());
    return Error::Internal;
  }
  // for debugging purpose
  const char* lib_name = et_pal_get_shared_library_name(operators.data());

  for (const auto& op : operators) {
    if (this->hasOpsFn(op.name_, {})) {
      ET_LOG(Error, "Re-registering %s. From: %s", op.name_, lib_name);
      return Error::InvalidArgument;
    }
    this->operators_table_[this->operatorRegSize_++] = op;
  }
  ET_LOG(
      Debug,
      "Successfully registered all ops from shared library: %s",
      lib_name);

  return Error::Ok;
}

Error register_kernels(const ArrayRef<Kernel>& kernels) {
  Error success = getOperatorRegistry().register_kernels(kernels);
  if (success == Error::InvalidArgument || success == Error::Internal) {
    ET_CHECK_MSG(
        false,
        "Kernel registration failed with error %" PRIu32
        ", see error log for details.",
        success);
  }
  return success;
}

Error OperatorRegistry::register_kernels(const ArrayRef<Kernel>& kernels) {
  // Operator registration happens in static initialization time when PAL init
  // may or may not happen already. Here we are assuming et_pal_init() doesn't
  // have any side effect even if falled multiple times.
  ::et_pal_init();

  // for debugging purpose
  const char* lib_name = et_pal_get_shared_library_name(kernels.data());

  for (const auto& kernel : kernels) {
    bool result = false;
    for (size_t idx = 0; idx < operatorRegSize_; idx++) {
      if (strcmp(operators_table_[idx].name_, kernel.name_) == 0) {
        // re-registering kernel
        if (operators_table_[idx].contains(kernel.kernel_key_)) {
          ET_LOG(Error, "Re-registering %s. From: %s", kernel.name_, lib_name);
          return Error::InvalidArgument;
        }
        result = operators_table_[idx].register_kernel(kernel);
        // more kernels than what's supported
        if (!result) {
          ET_LOG(
              Error,
              "More than %d kernels are being registered to %s",
              kMaxNumOfKernelPerOp,
              kernel.name_);
          return Error::Internal;
        }
      }
    }
    // no such operator in registry yet, create a new one
    if (!result) {
      Operator op = Operator(kernel.name_, kernel.kernel_key_, kernel.op_);
      Error err = register_operators({op});
      if (err != Error::Ok) {
        return err;
      }
    }
  }

  return Error::Ok;
}

bool hasOpsFn(const char* name, ArrayRef<TensorMeta> kernel_key) {
  return getOperatorRegistry().hasOpsFn(name, kernel_key);
}

static void make_kernel_key_string(ArrayRef<TensorMeta> key, char* buf) {
  if (key.empty()) {
    // If no tensor is present in an op, kernel key does not apply
    *buf = 0xff;
    return;
  }
  strncpy(buf, "v0/", 3);
  buf += 3;
  for (size_t i = 0; i < key.size(); i++) {
    auto& meta = key[i];
    *buf = (char)meta.dtype_;
    buf += 1;
    *buf = ';';
    buf += 1;
    memcpy(buf, (char*)meta.dim_order_.data(), meta.dim_order_.size());
    buf += meta.dim_order_.size();
    *buf = (i < (key.size() - 1)) ? '|' : 0xff;
    buf += 1;
  }
}

constexpr int BUF_SIZE = 307;

bool OperatorRegistry::hasOpsFn(
    const char* name,
    ArrayRef<TensorMeta> meta_list) {
  for (size_t idx = 0; idx < this->operatorRegSize_; idx++) {
    if (strcmp(this->operators_table_[idx].name_, name) == 0) {
      if (this->operators_table_[idx].has_fallback()) {
        return true;
      }
    }
  }

  if (meta_list.empty()) {
    // If no tensor is present (fallback is required) but no fallback is
    // available, return false
    return false;
  }

  char buf[BUF_SIZE];
  make_kernel_key_string(meta_list, buf);
  KernelKey kernel_key = KernelKey(buf);
  for (size_t idx = 0; idx < this->operatorRegSize_; idx++) {
    if (strcmp(this->operators_table_[idx].name_, name) == 0) {
      if (this->operators_table_[idx].contains(kernel_key)) {
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
  char buf[BUF_SIZE];
  make_kernel_key_string(meta_list, buf);
  KernelKey kernel_key = KernelKey(buf);
  for (size_t idx = 0; idx < this->operatorRegSize_; idx++) {
    if (strcmp(this->operators_table_[idx].name_, name) == 0) {
      return this->operators_table_[idx].find_or_fallback(kernel_key);
    }
  }
  ET_CHECK_MSG(false, "operator '%s' not found.", name);
}

ArrayRef<Operator> getOpsArray() {
  return getOperatorRegistry().getOpsArray();
}

ArrayRef<Operator> OperatorRegistry::getOpsArray() {
  return ArrayRef<Operator>(this->operators_table_, this->operatorRegSize_);
}

} // namespace executor
} // namespace torch
