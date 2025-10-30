// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/backends/cuda/runtime/slim/core/SlimTensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace standalone::slim {

// Helper to convert ExecuTorch ScalarType to standalone::c10::ScalarType
inline standalone::c10::ScalarType etensor_to_slim_dtype(
    executorch::aten::ScalarType et_dtype) {
  switch (et_dtype) {
  case executorch::aten::ScalarType::Float:
    return standalone::c10::ScalarType::Float;
  case executorch::aten::ScalarType::Double:
    return standalone::c10::ScalarType::Double;
  case executorch::aten::ScalarType::Half:
    return standalone::c10::ScalarType::Half;
  case executorch::aten::ScalarType::BFloat16:
    return standalone::c10::ScalarType::BFloat16;
  case executorch::aten::ScalarType::Long:
    return standalone::c10::ScalarType::Long;
  case executorch::aten::ScalarType::Int:
    return standalone::c10::ScalarType::Int;
  case executorch::aten::ScalarType::Short:
    return standalone::c10::ScalarType::Short;
  case executorch::aten::ScalarType::Char:
    return standalone::c10::ScalarType::Char;
  case executorch::aten::ScalarType::Byte:
    return standalone::c10::ScalarType::Byte;
  case executorch::aten::ScalarType::Bool:
    return standalone::c10::ScalarType::Bool;
  default:
    STANDALONE_CHECK(false, "Unsupported ETensor dtype for SlimTensor");
  }
}

// Extended SlimTensor class with ETensor constructor
class SlimTensorWithETensor : public SlimTensor {
public:
  using SlimTensor::SlimTensor;

  // Constructor from ETensor - does NOT take ownership of the data
  explicit SlimTensorWithETensor(
      executorch::runtime::etensor::Tensor *etensor)
      : SlimTensor() {
    STANDALONE_CHECK(etensor != nullptr,
                     "Cannot create SlimTensor from null ETensor");

    // Get ETensor properties
    auto et_sizes = etensor->sizes();
    auto et_strides = etensor->strides();
    auto et_dtype = etensor->scalar_type();
    void *data_ptr = etensor->mutable_data_ptr();

    // Convert sizes and strides to int64_t vectors
    std::vector<int64_t> sizes(et_sizes.begin(), et_sizes.end());
    std::vector<int64_t> strides(et_strides.begin(), et_strides.end());

    // Convert dtype
    auto slim_dtype = etensor_to_slim_dtype(et_dtype);

    // Create a non-owning storage that wraps the ETensor's data
    // We use CPU device since ETensor doesn't carry device info
    // For CUDA tensors, the caller should handle device appropriately
    Storage storage(new MaybeOwningStorage(
        data_ptr, etensor->nbytes(), CPU_DEVICE,
        false // non-owning - ETensor manages the memory
        ));

    // Initialize the SlimTensor with the wrapped storage
    *this = SlimTensor(std::move(storage), sizes, strides, slim_dtype, 0);
  }
};

} // namespace standalone::slim
