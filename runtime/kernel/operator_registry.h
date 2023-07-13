#pragma once

#include <cstring>

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/function_ref.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>
// Debug switch for operator registry
#if defined(ET_OP_REGISTRY_DEBUG)
#include <ostream>
#endif

namespace torch {
namespace executor {

class KernelRuntimeContext; // Forward declaration
using RuntimeContext = KernelRuntimeContext; // TODO(T147221312): Remove
using OpFunction = FunctionRef<void(KernelRuntimeContext&, EValue**)>;

/**
 * Dtype and dim order metadata for a Tensor argument to an operator.
 * Used by the Executor to hold the tensor metadata info.
 */
struct TensorMeta {
  exec_aten::ScalarType dtype_;
  ArrayRef<exec_aten::DimOrderType> dim_order_;

  TensorMeta() = default;
  TensorMeta(ScalarType dtype, ArrayRef<exec_aten::DimOrderType> order)
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
    for (int i = 0; i < dim_order_.size(); i++) {
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
 * operator. If `is_fallback_` is true, it means this kernel can be used as a
 * fallback, if false, it means this kernel can only be used if all the
 * `TensorMeta` are matched. Fallback means this kernel will be used for
 * all input tensor dtypes and dim orders, if the specialized kernel is not
 * registered.
 *
 * The format of a kernel key data is a string:
 *                              "v<version>/<tensor_meta>|<tensor_meta>...\xff"
 * Size: Up to 307               1    1    1     (18     +1) * 16
 *           Assuming max number of tensors is 16               ^
 * Version is v0 for now
 * Example: v0/0x07;0x00 0x01 0x02 0x03 \xff
 * The kernel key has only one tensor: a double tensor with dimension 0, 1, 2, 3
 *
 * The string is a byte array and contains non-printable characters. It must
 * be terminated with a '\xff' so 0xff cannot be a scalar type.
 *
 * Each tensor_meta has the following format: "<dtype>;<dim_order...>"
 * Size: Up to 18                                 1   1    16
 * Assuming that the max number of dims is 16              ^
 * Example: 0x07;0x00 0x01 0x02 0x03 for [double; 0, 1, 2, 3]
 *
 * IMPORTANT:
 * Users should not construct a kernel key manually. Instead, it should be
 * generated from kernel yaml.
 */
struct KernelKey {
 public:
  KernelKey() : is_fallback_(true) {}

  /* implicit */ KernelKey(const char* kernel_key_data)
      : kernel_key_data_(kernel_key_data), is_fallback_(false) {}

  constexpr static char TERMINATOR = 0xff;

  bool operator==(const KernelKey& other) const {
    return this->equals(other);
  }

  bool operator!=(const KernelKey& other) const {
    return !this->equals(other);
  }

  bool equals(const KernelKey& other) const {
    if (is_fallback_ != other.is_fallback_) {
      return false;
    }
    if (is_fallback_) {
      return true;
    }
    size_t i;
    for (i = 0; kernel_key_data_[i] != TERMINATOR &&
         other.kernel_key_data_[i] != TERMINATOR;
         i++) {
      if (kernel_key_data_[i] != other.kernel_key_data_[i]) {
        return false;
      }
    }
    return kernel_key_data_[i] == TERMINATOR &&
        other.kernel_key_data_[i] == TERMINATOR;
  }

  bool is_fallback() const {
    return is_fallback_;
  }

#if defined(ET_OP_REGISTRY_DEBUG)
  friend std::ostream& operator<<(std::ostream& os, const KernelKey& key) {
    os << key.kernel_key_data_ << std::endl;
    return os;
  }
#endif

 private:
  const char* kernel_key_data_ = nullptr;
  bool is_fallback_;
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

constexpr uint32_t kOperatorTableMaxSize = 200;
constexpr uint32_t kMaxNumOfKernelPerOp = 8;
constexpr uint32_t kMaxNumOfKernels =
    kOperatorTableMaxSize * kMaxNumOfKernelPerOp;

/**
 * Struct that represents an operator at runtime. This object and the `Operator`
 * field in the program should be 1-to-1 mapping. During static initialization,
 * all kernels will be registered from the generated C++ code. Then during the
 * kernel resolution step in runtime initialization, the target kernel will be
 * looked up and stored along with `Chain`.
 */
struct Operator {
 public:
  const char* name_;
  explicit Operator(const char* name) : name_(name), num_kernels_(0) {}

  // constructor that takes a kernel with its kernel key.
  explicit Operator(const char* name, KernelKey key, OpFunction func)
      : name_(name), num_kernels_(1) {
    kernels_[0] = Kernel(name, key, func);
  }
  explicit Operator(const char* name, OpFunction func)
      : name_(name), num_kernels_(1) {
    kernels_[0] = Kernel(name, {}, func);
  }
  Operator() {}

  // check if this operator contains a kernel with a particular kernel key.
  bool contains(KernelKey key) const {
    for (auto i = 0; i < num_kernels_; i++) {
      if (kernels_[i].kernel_key_ == key) {
        return true;
      }
    }
    return false;
  }

  // returns an `OpFunction` from either a kernel key match, or fallback kernel
  // if not matched.
  const OpFunction& find_or_fallback(KernelKey key) const {
    int32_t fallback_index = -1;
    for (auto i = 0; i < num_kernels_; i++) {
      if (kernels_[i].kernel_key_ == key) {
        return kernels_[i].op_;
      }
      if (kernels_[i].kernel_key_.is_fallback()) {
        fallback_index = i;
      }
    }
    if (fallback_index != -1) {
      return kernels_[fallback_index].op_;
    }
    ET_CHECK_MSG(false, "kernel key not found.");
  }

  bool has_fallback() const {
    return contains({});
  }

  bool register_kernel(Kernel kernel) {
    if (num_kernels_ == kMaxNumOfKernelPerOp) {
      return false;
    }
    kernels_[num_kernels_++] = kernel;
    return true;
  }

 private:
  Kernel kernels_[kMaxNumOfKernelPerOp];
  uint32_t num_kernels_;
};

/**
 * See OperatorRegistry::hasOpsFn()
 */
bool hasOpsFn(const char* name, ArrayRef<TensorMeta> meta_list = {});

/**
 * See OperatorRegistry::getOpsFn()
 */
const OpFunction& getOpsFn(
    const char* name,
    ArrayRef<TensorMeta> meta_list = {});

/**
 * See OperatorRegistry::getOpsArray()
 */
ArrayRef<Operator> getOpsArray();

/**
 * DEPRECATED: Use register_kernels() instead.
 * See OperatorRegistry::register_operators(). Notice that the returned Error
 * object should be handled internally and the reason for keep returning is to
 * satisfy the requirement to run this in static initialization time.
 */
__ET_NODISCARD Error register_operators(const ArrayRef<Operator>&);

/**
 * See OperatorRegistry::register_kernels(). Notice that the returned Error
 * object should be handled internally and the reason for keep returning is to
 * satisfy the requirement to run this in static initialization time.
 */
__ET_NODISCARD Error register_kernels(const ArrayRef<Kernel>&);

struct OperatorRegistry {
 public:
  OperatorRegistry() : operatorRegSize_(0) {}

  /**
   * DEPRECATED: Use register_kernels() instead. TODO: (larryliu) Remove.
   * Registers the Operator object (which may contain one or more function
   * references) so that it could be called via the name during the runtime.
   * WARNING: only use this when we are confident that there are no duplicates
   * in Operator name.
   * @param[in] operators Operator object
   * @retval Error code representing whether registration was successful.
   */
  __ET_NODISCARD Error register_operators(const ArrayRef<Operator>&);

  /**
   * Registers the Kernels object (i.e. string name and function reference
   * pair). The kernels will be merged into Operators based on the op name.
   *
   * @param[in] kernels Kernel object
   * @retval Error code representing whether registration was successful.
   */
  __ET_NODISCARD Error register_kernels(const ArrayRef<Kernel>&);

  /**
   * Checks whether an operator with a given name and TensorMeta list.
   * When TensorMeta is empty, it means this op does not have specialized
   * kernels, so it checks whether it has any fallback kernels.
   */
  bool hasOpsFn(const char* name, ArrayRef<TensorMeta> meta_list);

  /**
   * Get the operator with a given name and TensorMeta list
   */
  const OpFunction& getOpsFn(const char* name, ArrayRef<TensorMeta> meta_list);

  /**
   * Return all registered operators.
   */
  ArrayRef<Operator> getOpsArray();

 private:
  Operator operators_table_[kOperatorTableMaxSize];
  uint32_t operatorRegSize_;
};

} // namespace executor
} // namespace torch
