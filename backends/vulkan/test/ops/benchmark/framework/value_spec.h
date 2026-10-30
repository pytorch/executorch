// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

using namespace vkcompute;

//
// ValueSpec class
//

enum class SpecType { Tensor, IntList, Int, Float, Bool, String };

// Data generation types
enum class DataGenType {
  FIXED,
  RANDOM,
  RANDOM_SCALES,
  RANDINT,
  RANDINT8,
  RANDINT4,
  ONES,
  ONES_INT4,
  ZEROS
};

// Value specification struct
struct ValueSpec {
  std::vector<int64_t> sizes;
  vkapi::ScalarType dtype;
  utils::GPUMemoryLayout memory_layout;
  utils::StorageType storage_type;
  SpecType spec_type;
  DataGenType data_gen_type;
  bool is_constant_tensor;
  bool is_none_flag;
  bool is_int4_tensor;
  std::string string_data;

  ValueSpec(
      const std::vector<int64_t>& sizes,
      vkapi::ScalarType dtype,
      utils::StorageType storage_type = utils::kTexture3D,
      utils::GPUMemoryLayout memory_layout = utils::kWidthPacked)
      : sizes(sizes),
        dtype(dtype),
        memory_layout(memory_layout),
        storage_type(storage_type),
        spec_type(SpecType::Tensor),
        data_gen_type(DataGenType::ZEROS),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(false) {
    // Data generation is deferred until first access (any data getter or
    // ensure_data_generated() triggers it).
  }

  // Constructor for tensor with custom data generation type
  ValueSpec(
      const std::vector<int64_t>& sizes,
      vkapi::ScalarType dtype,
      utils::StorageType storage_type,
      utils::GPUMemoryLayout memory_layout,
      DataGenType data_gen_type)
      : sizes(sizes),
        dtype(dtype),
        memory_layout(memory_layout),
        storage_type(storage_type),
        spec_type(SpecType::Tensor),
        data_gen_type(data_gen_type),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(false) {
    // Data generation is deferred until first access (any data getter or
    // ensure_data_generated() triggers it).
  }

  // Constructor for single int
  ValueSpec(int32_t value)
      : sizes({1}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Int),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(true) {
    data_->int32_data.push_back(value);
  }

  // Constructor for single float
  ValueSpec(float value)
      : sizes({1}),
        dtype(vkapi::kFloat),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Float),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(true) {
    data_->float_data.push_back(value);
  }

  // Constructor for single bool
  ValueSpec(bool value)
      : sizes({1}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Bool),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(true) {
    data_->int32_data.push_back(value ? 1 : 0);
  }

  // Constructor for int list
  ValueSpec(const std::vector<int32_t>& values)
      : sizes({static_cast<int64_t>(values.size())}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::IntList),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(true) {
    data_->int32_data = values;
  }

  // Factory method for string (avoids ambiguity with vector constructor)
  static ValueSpec make_string(const std::string& value) {
    ValueSpec spec;
    spec.sizes = {1};
    spec.dtype = vkapi::kInt;
    spec.memory_layout = utils::kWidthPacked;
    spec.storage_type = utils::kTexture3D;
    spec.spec_type = SpecType::String;
    spec.data_gen_type = DataGenType::FIXED;
    spec.is_constant_tensor = false;
    spec.is_none_flag = false;
    spec.is_int4_tensor = false;
    spec.data_generated_ = true;
    spec.string_data = value;
    return spec;
  }

  // Default constructor
  ValueSpec()
      : dtype(vkapi::kFloat),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Tensor),
        data_gen_type(DataGenType::ZEROS),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        data_generated_(false) {}

  int64_t numel() const;
  size_t nbytes() const;
  std::string to_string() const;

  bool is_tensor() const {
    return spec_type == SpecType::Tensor;
  }
  bool is_int_list() const {
    return spec_type == SpecType::IntList;
  }
  bool is_int() const {
    return spec_type == SpecType::Int;
  }
  bool is_float() const {
    return spec_type == SpecType::Float;
  }
  bool is_bool() const {
    return spec_type == SpecType::Bool;
  }
  bool is_string() const {
    return spec_type == SpecType::String;
  }

  int32_t get_int_value() const {
    ensure_data_generated();
    return data_->int32_data.empty() ? 0 : data_->int32_data[0];
  }
  float get_float_value() const {
    ensure_data_generated();
    return data_->float_data.empty() ? 0.0f : data_->float_data[0];
  }
  bool get_bool_value() const {
    ensure_data_generated();
    return data_->int32_data.empty() ? false : (data_->int32_data[0] != 0);
  }
  const std::string& get_string_value() const {
    return string_data;
  }
  const std::vector<int32_t>& get_int_list() const {
    ensure_data_generated();
    return data_->int32_data;
  }
  const std::vector<int64_t>& get_tensor_sizes() const {
    return sizes;
  }

  // References and pointers into tensor data must not be held across any other
  // access to the same spec: a mutable access may detach the shared payload,
  // leaving a previously returned reference bound to the old payload. Consume
  // immediately.
  const std::vector<float>& get_float_data() const {
    ensure_data_generated();
    return data_->float_data;
  }
  const std::vector<int32_t>& get_int32_data() const {
    ensure_data_generated();
    return data_->int32_data;
  }
  const std::vector<uint16_t>& get_half_data() const {
    ensure_data_generated();
    return data_->half_data;
  }
  const std::vector<int8_t>& get_int8_data() const {
    ensure_data_generated();
    return data_->int8_data;
  }
  const std::vector<uint8_t>& get_uint8_data() const {
    ensure_data_generated();
    return data_->uint8_data;
  }

  std::vector<float>& get_float_data() {
    ensure_data_generated();
    ensure_unique_data();
    return data_->float_data;
  }
  std::vector<int32_t>& get_int32_data() {
    ensure_data_generated();
    ensure_unique_data();
    return data_->int32_data;
  }
  std::vector<uint16_t>& get_half_data() {
    ensure_data_generated();
    ensure_unique_data();
    return data_->half_data;
  }
  std::vector<int8_t>& get_int8_data() {
    ensure_data_generated();
    ensure_unique_data();
    return data_->int8_data;
  }
  std::vector<uint8_t>& get_uint8_data() {
    ensure_data_generated();
    ensure_unique_data();
    return data_->uint8_data;
  }

  const std::vector<float>& get_ref_float_data() const {
    return reference_data_->float_data;
  }
  const std::vector<int32_t>& get_ref_int32_data() const {
    return reference_data_->int32_data;
  }
  const std::vector<uint16_t>& get_ref_half_data() const {
    return reference_data_->half_data;
  }
  const std::vector<int8_t>& get_ref_int8_data() const {
    return reference_data_->int8_data;
  }
  const std::vector<uint8_t>& get_ref_uint8_data() const {
    return reference_data_->uint8_data;
  }

  std::vector<float>& get_ref_float_data() {
    ensure_unique_reference_data();
    return reference_data_->float_data;
  }
  std::vector<int32_t>& get_ref_int32_data() {
    ensure_unique_reference_data();
    return reference_data_->int32_data;
  }
  std::vector<uint16_t>& get_ref_half_data() {
    ensure_unique_reference_data();
    return reference_data_->half_data;
  }
  std::vector<int8_t>& get_ref_int8_data() {
    ensure_unique_reference_data();
    return reference_data_->int8_data;
  }
  std::vector<uint8_t>& get_ref_uint8_data() {
    ensure_unique_reference_data();
    return reference_data_->uint8_data;
  }

  void resize_data(size_t new_size);
  void* get_mutable_data_ptr();
  float get_element(size_t index) const;

  // Data generation methods for deferred generation and caching.
  //
  // ValueSpec is not thread-safe: lazy materialization and copy-on-write
  // detach mutate shared state from const methods. Test cases are built and
  // executed on a single thread.
  //
  // Implicit materialization (any data getter, resize_data) consumes the
  // global seed counter. Callers needing deterministic data must call
  // ensure_data_generated(explicit_seed) before any other access; a later
  // seeded call is a no-op once data is generated.
  bool is_data_generated() const {
    return data_generated_;
  }
  void ensure_data_generated(int seed = -1) const;
  void share_data_from(const ValueSpec& other);
  void share_reference_from(const ValueSpec& other);

  // Set/get constant flag
  bool is_constant() const {
    return is_constant_tensor;
  }
  void set_constant(bool is_constant) {
    is_constant_tensor = is_constant;
  }

  // Set/get none flag
  bool is_none() const {
    return is_none_flag;
  }

  void set_none(bool is_none) {
    is_none_flag = is_none;
  }

  // Set/get int4 flag
  bool is_int4() const {
    return is_int4_tensor;
  }
  void set_int4(bool is_int4) {
    is_int4_tensor = is_int4;
  }

  const void* get_data_ptr() const;

  // Correctness checking against reference data
  // Returns true if computed data matches reference data within tolerance
  // Only validates float tensors as specified in requirements
  bool validate_against_reference(
      float abs_tolerance = 2e-3f,
      float rel_tolerance = 1e-3f) const;

 private:
  struct TensorData {
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<uint16_t> half_data;
    std::vector<int8_t> int8_data;
    std::vector<uint8_t> uint8_data;
  };

  void ensure_unique_data() const;
  void ensure_unique_reference_data() const;
  void generate_tensor_data(int seed = -1) const;

  mutable bool data_generated_ = false;
  mutable std::shared_ptr<TensorData> data_ = std::make_shared<TensorData>();
  mutable std::shared_ptr<TensorData> reference_data_ =
      std::make_shared<TensorData>();
};

void print_valuespec_data(
    const ValueSpec& spec,
    const std::string& name = "ValueSpec",
    const bool print_ref_data = false,
    size_t max_elements = 20,
    int precision = 6);

// Half-precision conversion utilities
uint16_t float_to_half(float value);
float half_to_float(uint16_t half_val);

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
