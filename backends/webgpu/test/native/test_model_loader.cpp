/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runner/webgpu_model_loader.h>

#include <gtest/gtest.h>

namespace executorch::backends::webgpu {
namespace {

TEST(WebGPUModelLoaderTest, RejectsEmptyProgramPath) {
  WebGPUModelLoadSpec spec;
  spec.required_methods = {"forward"};
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), runtime::Error::InvalidArgument);
}

TEST(WebGPUModelLoaderTest, RejectsEmptyRequiredMethods) {
  WebGPUModelLoadSpec spec;
  spec.pte_path = "not-opened.pte";
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), runtime::Error::InvalidArgument);
}

TEST(WebGPUModelLoaderTest, RejectsDuplicateRequiredMethod) {
  WebGPUModelLoadSpec spec;
  spec.pte_path = "not-opened.pte";
  spec.required_methods = {"forward", "forward"};
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), runtime::Error::InvalidArgument);
}

TEST(WebGPUModelLoaderTest, RejectsEmptyPtdPath) {
  WebGPUModelLoadSpec spec;
  spec.pte_path = "not-opened.pte";
  spec.ptd_paths = {""};
  spec.required_methods = {"forward"};
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), runtime::Error::InvalidArgument);
}

TEST(WebGPUModelLoaderTest, RejectsDuplicatePtdPath) {
  WebGPUModelLoadSpec spec;
  spec.pte_path = "not-opened.pte";
  spec.ptd_paths = {"weights.ptd", "weights.ptd"};
  spec.required_methods = {"forward"};
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), runtime::Error::InvalidArgument);
}

TEST(WebGPUModelLoaderTest, InvalidProgramFailsWithoutPublishingModule) {
  WebGPUModelLoadSpec spec;
  spec.pte_path = "does-not-exist.pte";
  spec.ptd_paths = {"first.ptd", "second.ptd"};
  spec.required_methods = {"forward"};
  spec.load_mode = extension::Module::LoadMode::Mmap;
  const auto result = load_webgpu_model(std::move(spec));
  EXPECT_FALSE(result.ok());
}

} // namespace
} // namespace executorch::backends::webgpu
