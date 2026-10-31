/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <algorithm>
#include <type_traits>

#import "coreai_fake_loader.h"
#include "coreai_source_fixture.h"

namespace executorch::backends::coreai::testing {

class DelegateFixture final {
 public:
  DelegateFixture(
      const runtime::BackendInterface* backend,
      ScopedFakeBridgeState& bridge,
      NSDictionary* manifest = manifest_dict());
  ~DelegateFixture();
  DelegateFixture(const DelegateFixture&) = delete;
  DelegateFixture& operator=(const DelegateFixture&) = delete;

  runtime::Error init(NSString* root = nil);
  runtime::Error init_with_options(
      runtime::Span<const runtime::BackendOption> options,
      runtime::MemoryAllocator* allocator = nullptr);
  void reset();
  runtime::Error execute(runtime::Span<runtime::EValue*> args);

  TestData data;
  TestDirectory storage;
  int freed = 0;
  NSData* json;
  runtime::FreeableBuffer processed;
  runtime::DelegateHandle* handle = nullptr;

 private:
  const runtime::BackendInterface* backend_;
  ScopedFakeBridgeState& bridge_;
};

struct FloatTensors final {
  float input[2] = {3, 7};
  float output[2] = {-1, -1};
  executorch::aten::TensorImpl::SizesType input_size[1] = {2};
  executorch::aten::TensorImpl::SizesType output_size[1] = {2};
  executorch::aten::TensorImpl::DimOrderType order[1] = {0};
  executorch::aten::TensorImpl::StridesType input_stride[1] = {1};
  executorch::aten::TensorImpl::StridesType output_stride[1] = {1};
  executorch::aten::TensorImpl input_impl{
      executorch::aten::ScalarType::Float,
      1,
      input_size,
      input,
      order,
      input_stride};
  executorch::aten::TensorImpl output_impl{
      executorch::aten::ScalarType::Float,
      1,
      output_size,
      output,
      order,
      output_stride,
      runtime::TensorShapeDynamism::DYNAMIC_BOUND};
  runtime::EValue in{executorch::aten::Tensor(&input_impl)};
  runtime::EValue out{executorch::aten::Tensor(&output_impl)};
  runtime::EValue* args[2] = {&in, &out};
};

NSMutableDictionary* multiple_io_manifest();

template <typename T>
struct RangeTensors final {
  static constexpr executorch::aten::ScalarType dtype = std::is_same_v<T, float>
      ? executorch::aten::ScalarType::Float
      : executorch::aten::ScalarType::Half;
  T input[12] = {};
  T output[12] = {};
  executorch::aten::TensorImpl::SizesType input_shape[2] = {4, 2};
  executorch::aten::TensorImpl::SizesType output_shape[2] = {4, 2};
  executorch::aten::TensorImpl::DimOrderType order[2] = {0, 1};
  executorch::aten::TensorImpl::StridesType input_strides[2] = {2, 1};
  executorch::aten::TensorImpl::StridesType output_strides[2] = {2, 1};
  executorch::aten::TensorImpl input_impl{
      dtype,
      2,
      input_shape,
      input + 2,
      order,
      input_strides,
      runtime::TensorShapeDynamism::DYNAMIC_BOUND};
  executorch::aten::TensorImpl output_impl{
      dtype,
      2,
      output_shape,
      output + 2,
      order,
      output_strides,
      runtime::TensorShapeDynamism::DYNAMIC_BOUND};
  runtime::EValue in{executorch::aten::Tensor(&input_impl)};
  runtime::EValue out{executorch::aten::Tensor(&output_impl)};
  runtime::EValue* args[2] = {&in, &out};

  RangeTensors() {
    for (size_t i = 0; i < 12; ++i) {
      input[i] = static_cast<T>(i + 1);
    }
    fill_output();
  }

  void fill_output() {
    std::fill_n(output, 12, static_cast<T>(99));
  }
};

} // namespace executorch::backends::coreai::testing
