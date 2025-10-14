/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple audio struct.

#pragma once
#include <executorch/runtime/platform/compiler.h>
#include <cstdint>
#include <vector>

#include <executorch/extension/tensor/tensor.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * Audio inputs as a raw audio tensor, for use when the audio processing
 * into a mel spectrogram is baked into the audio encoder with torch.export.
 */
struct ET_EXPERIMENTAL RawAudio {
  std::vector<uint8_t> data;
  int32_t batch_size;
  int32_t n_channels; // For mono, use n_channels = 1.
  int32_t n_samples;
};

/**
 * Pre-processed audio inputs, ready to feed directly into an audio encoder.
 *
 * The data can be either uint8_t or float. If the audio has gone through a Mel
 * transform, we expect the data type to be float (i.e., std::vector<float>), as
 * Mel spectrograms are typically represented as floating point values. For raw
 * or quantized audio, uint8_t may be used instead.
 */
class ET_EXPERIMENTAL Audio final {
 public:
  // Constructor for uint8_t data
  Audio(
      std::vector<uint8_t>&& data,
      int32_t batch_size,
      int32_t n_bins,
      int32_t n_frames)
      : Audio(make_tensor_ptr(
            {batch_size, n_bins, n_frames},
            std::move(data),
            executorch::aten::ScalarType::Byte)) {}

  // Constructor for float data
  Audio(
      std::vector<float>&& data,
      int32_t batch_size,
      int32_t n_bins,
      int32_t n_frames)
      : Audio(make_tensor_ptr({batch_size, n_bins, n_frames}, std::move(data))) {}

  explicit Audio(
      executorch::extension::TensorPtr tensor) : tensor_(std::move(tensor)) {
    ET_CHECK_MSG(tensor_, "Null tensor");
    ET_CHECK_MSG(tensor_->dim() == 3, "Invalid tensor rank");
  }

  // Type checkers
  bool is_uint8() const {
    return tensor_->scalar_type() == ::executorch::aten::ScalarType::Byte;
  }

  bool is_float() const {
    return tensor_->scalar_type() == ::executorch::aten::ScalarType::Float;
  }

  // Data access
  const uint8_t* uint8_data() const {
    ET_DCHECK_MSG(is_uint8(), "Dtype is not uint8");
    return tensor_->const_data_ptr<uint8_t>();
  }

  const float* float_data() const {
    ET_DCHECK_MSG(is_float(), "Dtype is not float");
    return tensor_->const_data_ptr<float>();
  }

  int32_t get_batch_size() const {
    return tensor_->size(0);
  }
  int32_t get_n_bins() const {
    return tensor_->size(1);
  }
  int32_t get_n_frames() const {
    return tensor_->size(2);
  }
  /**
   * Convert the audio data to a TensorPtr, with optional batch dimension.
   * The tensor will have shape (batch_size, n_bins, n_frames) or (1,
   * batch_size, n_bins, n_frames) if with_batch is true.
   */
  executorch::extension::TensorPtr tensor(
      bool with_batch = false) const {
    if (with_batch) {
      return make_tensor_ptr(
          *tensor_,
          {1,
           static_cast<executorch::aten::SizesType>(tensor_->size(0)),
           static_cast<executorch::aten::SizesType>(tensor_->size(1)),
           static_cast<executorch::aten::SizesType>(tensor_->size(2))});
    }
    return tensor_;
  }

 private:
  // Members
  executorch::extension::TensorPtr tensor_;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Audio;
} // namespace executor
} // namespace torch
