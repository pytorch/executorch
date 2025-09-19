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
#include <variant>
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
  // Default constructor
  Audio() : batch_size_(0), n_bins_(0), n_frames_(0) {}

  // Constructor for uint8_t data
  Audio(
      std::vector<uint8_t>&& data,
      int32_t batch_size,
      int32_t n_bins,
      int32_t n_frames)
      : data_(std::move(data)),
        batch_size_(batch_size),
        n_bins_(n_bins),
        n_frames_(n_frames) {
    ET_CHECK_MSG(
        data_.index() == 0 &&
            std::get<std::vector<uint8_t>>(data_).size() ==
                static_cast<size_t>(batch_size * n_bins * n_frames),
        "data.size() (%zu) does not match batch_size * n_bins * n_frames (%d)",
        std::get<std::vector<uint8_t>>(data_).size(),
        batch_size * n_bins * n_frames);
  }

  // Constructor for float data
  Audio(
      std::vector<float>&& data,
      int32_t batch_size,
      int32_t n_bins,
      int32_t n_frames)
      : data_(std::move(data)),
        batch_size_(batch_size),
        n_bins_(n_bins),
        n_frames_(n_frames) {
    ET_CHECK_MSG(
        data_.index() == 1 &&
            std::get<std::vector<float>>(data_).size() ==
                static_cast<size_t>(batch_size * n_bins * n_frames),
        "data.size() (%zu) does not match batch_size * n_bins * n_frames (%d)",
        std::get<std::vector<float>>(data_).size(),
        batch_size * n_bins * n_frames);
  }

  // Type checkers
  bool is_uint8() const {
    return std::holds_alternative<std::vector<uint8_t>>(data_);
  }

  bool is_float() const {
    return std::holds_alternative<std::vector<float>>(data_);
  }

  // Data access
  const std::vector<uint8_t>& get_uint8_data() const& {
    return std::get<std::vector<uint8_t>>(data_);
  }

  std::vector<uint8_t>& get_uint8_data() & {
    return std::get<std::vector<uint8_t>>(data_);
  }

  const std::vector<float>& get_float_data() const& {
    return std::get<std::vector<float>>(data_);
  }

  std::vector<float>& get_float_data() & {
    return std::get<std::vector<float>>(data_);
  }

  int32_t get_batch_size() const {
    return batch_size_;
  }
  int32_t get_n_bins() const {
    return n_bins_;
  }
  int32_t get_n_frames() const {
    return n_frames_;
  }
  /**
   * Convert the audio data to a TensorPtr, with optional batch dimension.
   * The tensor will have shape (batch_size, n_bins, n_frames) or (1,
   * batch_size, n_bins, n_frames) if with_batch is true.
   */
  executorch::runtime::Result<executorch::extension::TensorPtr> toTensor(
      bool with_batch = false) {
    const {
      std::vector<executorch::aten::SizesType> sizes = {
          get_batch_size(), get_n_bins(), get_n_frames()};
      if (with_batch) {
        sizes.insert(sizes.begin(), 1);
      }
      if (is_float()) {
        return executorch::extension::from_blob(
            const_cast<float*>(get_float_data().data()),
            sizes,
            ::executorch::aten::ScalarType::Float);
      } else if (is_uint8()) {
        return executorch::extension::from_blob(
            const_cast<uint8_t*>(get_uint8_data().data()),
            sizes,
            ::executorch::aten::ScalarType::Byte);
      }
      ET_LOG(
          Error,
          "Shouldn't reach here, audio data is not initialized with uint8_t or float vector.");
      return ::executorch::runtime::Error::NotSupported;
    }

   private:
    // Members
    std::variant<std::vector<uint8_t>, std::vector<float>> data_;
    int32_t batch_size_;
    int32_t n_bins_;
    int32_t n_frames_;
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
