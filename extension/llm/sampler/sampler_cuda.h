/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/sampler/sampler.h>

#ifdef CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cstddef>

namespace executorch {
namespace extension {
namespace llm {

struct SamplerWorkspace {
  SamplerWorkspace() = default;
  explicit SamplerWorkspace(size_t capacity) {
    reserve(capacity);
  }
  ~SamplerWorkspace() {
    release();
  }
  SamplerWorkspace(const SamplerWorkspace&) = delete;
  SamplerWorkspace& operator=(const SamplerWorkspace&) = delete;
  SamplerWorkspace(SamplerWorkspace&& other) noexcept {
    move_from(std::move(other));
  }
  SamplerWorkspace& operator=(SamplerWorkspace&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  inline void reserve(size_t capacity) {
    if (capacity > probs_capacity_) {
      resize_buffer(reinterpret_cast<void**>(&probs), capacity, sizeof(float));
      resize_buffer(
          reinterpret_cast<void**>(&indices), capacity, sizeof(int32_t));
      probs_capacity_ = capacity;
      indices_capacity_ = capacity;
    }
  }
  inline void ensure_temp_bytes(size_t bytes) {
    if (bytes > temp_capacity_) {
      resize_buffer(
          reinterpret_cast<void**>(&temp_storage), bytes, sizeof(uint8_t));
      temp_capacity_ = bytes;
    }
  }
  inline void ensure_argmax_capacity(size_t capacity) {
    if (capacity > argmax_capacity_) {
      resize_buffer(
          reinterpret_cast<void**>(&argmax_out), capacity, sizeof(uint64_t));
      argmax_capacity_ = capacity;
    }
  }

  size_t capacity() const {
    return probs_capacity_;
  }

  float* probs{nullptr};
  int32_t* indices{nullptr};
  uint8_t* temp_storage{nullptr};
  void* argmax_out{nullptr};

 private:
  inline void release() {
    if (probs) {
      cudaFree(probs);
      probs = nullptr;
    }
    if (indices) {
      cudaFree(indices);
      indices = nullptr;
    }
    if (temp_storage) {
      cudaFree(temp_storage);
      temp_storage = nullptr;
    }
    if (argmax_out) {
      cudaFree(argmax_out);
      argmax_out = nullptr;
    }
    probs_capacity_ = indices_capacity_ = argmax_capacity_ = temp_capacity_ = 0;
  }
  inline void move_from(SamplerWorkspace&& other) {
    probs = other.probs;
    indices = other.indices;
    temp_storage = other.temp_storage;
    argmax_out = other.argmax_out;
    probs_capacity_ = other.probs_capacity_;
    indices_capacity_ = other.indices_capacity_;
    argmax_capacity_ = other.argmax_capacity_;
    temp_capacity_ = other.temp_capacity_;

    other.probs = nullptr;
    other.indices = nullptr;
    other.temp_storage = nullptr;
    other.argmax_out = nullptr;
    other.probs_capacity_ = other.indices_capacity_ = other.argmax_capacity_ =
        other.temp_capacity_ = 0;
  }
  inline void resize_buffer(void** ptr, size_t new_cap, size_t elem_size) {
    if (*ptr) {
      cudaFree(*ptr);
      *ptr = nullptr;
    }
    cudaMalloc(ptr, new_cap * elem_size);
  }

  size_t probs_capacity_{0};
  size_t indices_capacity_{0};
  size_t argmax_capacity_{0};
  size_t temp_capacity_{0};
};

class CudaSampler {
 public:
  CudaSampler(
      int32_t vocab_size,
      float temperature,
      float topp,
      unsigned long long rng_seed,
      void* cuda_stream = nullptr,
      SamplerWorkspace* workspace = nullptr);

  CudaSampler(
      int32_t vocab_size,
      float temperature,
      void* cuda_stream = nullptr,
      SamplerWorkspace* workspace = nullptr);

  template <typename T>
  int32_t sample(T* logits);

 private:
  template <typename T>
  void apply_temperature_and_softmax_cuda(T* logits);
  template <typename T>
  int32_t sample_argmax_cuda(T* probabilities);
  template <typename T>
  int32_t sample_mult_cuda(T* probabilities, float coin);
  template <typename T>
  int32_t sample_topp_cuda(T* probabilities, float coin);

  template <typename T>
  void apply_temperature_and_softmax_host(T* logits);
  template <typename T>
  int32_t sample_argmax_host(T* probabilities);
  template <typename T>
  int32_t sample_mult_host(T* probabilities, float coin);
  template <typename T>
  int32_t sample_topp_host(T* probabilities, float coin);

  int32_t vocab_size_;
  float inv_temperature_;
  float topp_;
  unsigned long long rng_state_;
  void* cuda_stream_{nullptr};
  SamplerWorkspace* workspace_{nullptr};
};

} // namespace llm
} // namespace extension
} // namespace executorch

#endif // CUDA_AVAILABLE
