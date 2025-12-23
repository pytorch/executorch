/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/operator_registry.h>

#include <cinttypes>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/system.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

namespace {

// Maximum number of operators and their associated kernels that can be
// registered.
#ifdef MAX_KERNEL_NUM
constexpr uint32_t kMaxRegisteredKernels = MAX_KERNEL_NUM;
#else
constexpr uint32_t kMaxOperators = 250;
constexpr uint32_t kMaxKernelsPerOp = 8;
constexpr uint32_t kMaxRegisteredKernels = kMaxOperators * kMaxKernelsPerOp;
#endif

// Data that backs the kernel table. Since Kernel has a custom default
// constructor (implicitly, because it contains KernelKey, which has a custom
// ctor), some toolchains don't like having a global array of them: it would
// require constructing them at init time. Since we don't care about the values
// until we add each entry to the table, allocate static zeroed memory instead
// and point the table at it.
struct alignas(Kernel) KernelBuffer {
  uint8_t data[sizeof(Kernel)];
};

// @lint-ignore CLANGTIDY facebook-hte-CArray
KernelBuffer registered_kernels_data[kMaxRegisteredKernels];

/// Global table of registered kernels.
Kernel* registered_kernels = reinterpret_cast<Kernel*>(registered_kernels_data);

/// The number of kernels registered in the table.
size_t num_registered_kernels = 0;

// Registers the kernels, but may return an error.
Error register_kernels_internal(const Span<const Kernel> kernels) {
  // Operator registration happens in static initialization time before or after
  // PAL init, so call it here. It is safe to call multiple times.
  ::et_pal_init();

  if (kernels.size() + num_registered_kernels > kMaxRegisteredKernels) {
    ET_LOG(
        Error,
        "The total number of kernels to be registered is larger than the limit "
        "%" PRIu32 ". %" PRIu32
        " kernels are already registered and we're trying to register another "
        "%" PRIu32 " kernels.",
        kMaxRegisteredKernels,
        (uint32_t)num_registered_kernels,
        (uint32_t)kernels.size());
    ET_LOG(Error, "======== Kernels already in the registry: ========");
    for (size_t i = 0; i < num_registered_kernels; i++) {
      ET_LOG(Error, "%s", registered_kernels[i].name_);
      ET_LOG_KERNEL_KEY(registered_kernels[i].kernel_key_);
    }
    ET_LOG(Error, "======== Kernels being registered: ========");
    for (size_t i = 0; i < kernels.size(); i++) {
      ET_LOG(Error, "%s", kernels[i].name_);
      ET_LOG_KERNEL_KEY(kernels[i].kernel_key_);
    }
    return Error::RegistrationExceedingMaxKernels;
  }
  // for debugging purpose
  ET_UNUSED const char* lib_name =
      et_pal_get_shared_library_name(kernels.data());

  for (const auto& kernel : kernels) {
    // Linear search. This is fine if the number of kernels is small.
    for (size_t i = 0; i < num_registered_kernels; i++) {
      Kernel k = registered_kernels[i];
      if (strcmp(kernel.name_, k.name_) == 0 &&
          kernel.kernel_key_ == k.kernel_key_) {
        ET_LOG(Error, "Re-registering %s, from %s", k.name_, lib_name);
        ET_LOG_KERNEL_KEY(k.kernel_key_);
        return Error::RegistrationAlreadyRegistered;
      }
    }
    registered_kernels[num_registered_kernels++] = kernel;
  }
  ET_LOG(
      Debug,
      "Successfully registered all kernels from shared library: %s",
      lib_name);

  return Error::Ok;
}

} // namespace

// Registers the kernels, but panics if an error occurs. Always returns Ok.
Error register_kernels(const Span<const Kernel> kernels) {
  Error success = register_kernels_internal(kernels);
  if (success == Error::RegistrationAlreadyRegistered ||
      success == Error::RegistrationExceedingMaxKernels) {
    ET_CHECK_MSG(
        false,
        "Kernel registration failed with error %" PRIu32
        ", see error log for details.",
        static_cast<uint32_t>(success));
  }
  return success;
}

namespace {
/**
 * Writes `num` as a decimal string to `buf` and returns the number of bytes
 * written. Returns -1 if `buf` is too small or if `num` is not supported.
 */
int copy_char_as_number_to_buf(int num, char* buf, size_t buf_size) {
  if (num < 0) {
    return -1;
  }
  if (num < 10) {
    if (buf_size < 1) {
      return -1;
    }
    *buf = '0' + (char)num;
    return 1;
  }
  if (num < 100) {
    if (buf_size < 2) {
      return -1;
    }
    *buf++ = '0' + ((char)num) / 10;
    *buf = '0' + ((char)num) % 10;
    return 2;
  }
  return -1;
}
} // namespace

namespace internal {
Error make_kernel_key_string(
    Span<const TensorMeta> key,
    char* buf,
    size_t buf_size) {
  if (key.empty()) {
    // If no tensor is present in an op, kernel key does not apply.
    if (buf_size > 0) {
      buf[0] = '\0';
    }
    return Error::Ok;
  }

  // Reserve one byte for null terminator.
  if (buf_size < 1) {
    return Error::InvalidArgument;
  }
  buf_size -= 1;

  // Add prefix.
  if (buf_size < 3) {
    return Error::InvalidArgument;
  }
  memcpy(buf, "v1/", 3);
  buf += 3;
  buf_size -= 3;

  // Add tensor meta.
  for (size_t i = 0; i < key.size(); i++) {
    auto& meta = key[i];

    // Add dtype.
    int n = copy_char_as_number_to_buf((int)meta.dtype_, buf, buf_size);
    if (n < 0) {
      return Error::InvalidArgument;
    }
    buf += n;
    buf_size -= n;

    // Add separator between dtype and dim order.
    if (buf_size < 1) {
      return Error::InvalidArgument;
    }
    *buf++ = ';';
    buf_size -= 1;

    // Add dim order.
    for (size_t j = 0; j < meta.dim_order_.size(); j++) {
      n = copy_char_as_number_to_buf((int)meta.dim_order_[j], buf, buf_size);
      if (n < 0) {
        return Error::InvalidArgument;
      }
      buf += n;
      buf_size -= n;

      if (j < meta.dim_order_.size() - 1) {
        if (buf_size < 1) {
          return Error::InvalidArgument;
        }
        *buf++ = ',';
        buf_size -= 1;
      }
    }
    if (i < key.size() - 1) {
      if (buf_size < 1) {
        return Error::InvalidArgument;
      }
      *buf++ = '|';
      buf_size -= 1;
    }
  }
  *buf = '\0'; // Space for this was reserved above.
  return Error::Ok;
}
} // namespace internal

bool registry_has_op_function(
    const char* name,
    Span<const TensorMeta> meta_list) {
  return get_op_function_from_registry(name, meta_list).ok();
}

Result<OpFunction> get_op_function_from_registry(
    const char* name,
    Span<const TensorMeta> meta_list) {
  std::array<char, internal::kKernelKeyBufSize> key_string;
  Error err = internal::make_kernel_key_string(
      meta_list, key_string.data(), key_string.size());
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to make kernel key string");
    return err;
  }
  KernelKey kernel_key = KernelKey(key_string.data());

  int32_t fallback_idx = -1;
  for (size_t idx = 0; idx < num_registered_kernels; idx++) {
    if (strcmp(registered_kernels[idx].name_, name) == 0) {
      if (registered_kernels[idx].kernel_key_ == kernel_key) {
        return registered_kernels[idx].op_;
      }
      if (registered_kernels[idx].kernel_key_.is_fallback()) {
        fallback_idx = idx;
      }
    }
  }
  if (fallback_idx != -1) {
    return registered_kernels[fallback_idx].op_;
  }
  ET_LOG(Error, "kernel '%s' not found.", name);
  ET_LOG_TENSOR_META(meta_list);
  return Error::OperatorMissing;
}

Span<const Kernel> get_registered_kernels() {
  return {registered_kernels, num_registered_kernels};
}

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
