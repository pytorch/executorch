// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include <executorch/backends/native/extension/module/test/TestData.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <gtest/gtest.h>

namespace executorch::extension::native_module {
namespace {

void expect_same_tensor_signature(
    const runtime::TensorInfo& pte,
    const runtime::TensorInfo& ptn) {
  EXPECT_EQ(pte.scalar_type(), ptn.scalar_type());
  EXPECT_EQ(pte.nbytes(), ptn.nbytes());
  EXPECT_EQ(pte.name(), ptn.name());
  EXPECT_EQ(
      std::vector<int32_t>(pte.sizes().begin(), pte.sizes().end()),
      std::vector<int32_t>(ptn.sizes().begin(), ptn.sizes().end()));
  EXPECT_EQ(
      std::vector<uint8_t>(pte.dim_order().begin(), pte.dim_order().end()),
      std::vector<uint8_t>(ptn.dim_order().begin(), ptn.dim_order().end()));
}

// cppcheck-suppress-begin syntaxError
TEST(MethodMetaParityTest, MatchesPteTensorSubset) {
  const char* pte_path = std::getenv("ET_MODULE_ADD_PATH");
  ASSERT_NE(pte_path, nullptr);
  executorch_native_module_ptn_link_anchor();
  Module pte(pte_path);
  const std::vector<uint8_t> ptn_bytes =
      testing::make_tensor_package({"forward"}, /*num_inputs=*/2, {2, 2});
  Module ptn(
      std::make_unique<BufferDataLoader>(ptn_bytes.data(), ptn_bytes.size()));

  ASSERT_EQ(pte.load(), runtime::Error::Ok);
  ASSERT_EQ(ptn.load(), runtime::Error::Ok);
  const auto pte_meta = pte.method_meta("forward");
  const auto ptn_meta = ptn.method_meta("forward");
  ASSERT_TRUE(pte_meta.ok());
  ASSERT_TRUE(ptn_meta.ok());
  EXPECT_STREQ(pte_meta->name(), ptn_meta->name());
  // ModuleAdd.pte has two tensor inputs followed by one scalar input.
  ASSERT_EQ(pte_meta->num_inputs(), 3);
  ASSERT_EQ(ptn_meta->num_inputs(), 2);
  ASSERT_EQ(pte_meta->num_outputs(), ptn_meta->num_outputs());
  for (size_t i = 0; i < ptn_meta->num_inputs(); ++i) {
    const auto pte_tag = pte_meta->input_tag(i);
    const auto ptn_tag = ptn_meta->input_tag(i);
    ASSERT_TRUE(pte_tag.ok());
    ASSERT_TRUE(ptn_tag.ok());
    EXPECT_EQ(*pte_tag, *ptn_tag);
    const auto pte_tensor = pte_meta->input_tensor_meta(i);
    const auto ptn_tensor = ptn_meta->input_tensor_meta(i);
    ASSERT_TRUE(pte_tensor.ok());
    ASSERT_TRUE(ptn_tensor.ok());
    expect_same_tensor_signature(*pte_tensor, *ptn_tensor);
  }
  const auto extra_pte_input = pte_meta->input_tag(2);
  ASSERT_TRUE(extra_pte_input.ok());
  EXPECT_EQ(*extra_pte_input, runtime::Tag::Double);
  for (size_t i = 0; i < pte_meta->num_outputs(); ++i) {
    const auto pte_tag = pte_meta->output_tag(i);
    const auto ptn_tag = ptn_meta->output_tag(i);
    ASSERT_TRUE(pte_tag.ok());
    ASSERT_TRUE(ptn_tag.ok());
    EXPECT_EQ(*pte_tag, *ptn_tag);
    const auto pte_tensor = pte_meta->output_tensor_meta(i);
    const auto ptn_tensor = ptn_meta->output_tensor_meta(i);
    ASSERT_TRUE(pte_tensor.ok());
    ASSERT_TRUE(ptn_tensor.ok());
    expect_same_tensor_signature(*pte_tensor, *ptn_tensor);
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace executorch::extension::native_module
