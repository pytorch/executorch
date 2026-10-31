/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>

#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

namespace {

using ::executorch::extension::Module;
using ::executorch::extension::llm::get_max_prefill_chunk_size;
using ::executorch::runtime::Error;

// Fixtures from export_model_metadata.py: forward(tokens, input_pos) with
// get_max_seq_len = 8, the token dimension bounded at 7 or at 8.
// Returns nullptr if the fixture is missing or does not load.
std::unique_ptr<Module> load_fixture(const char* environment_variable) {
  const char* path = std::getenv(environment_variable);
  if (path == nullptr) {
    return nullptr;
  }
  auto module = std::make_unique<Module>(path);
  if (module->load() != Error::Ok) {
    return nullptr;
  }
  return module;
}

TEST(PrefillChunkSizeTest, LowersMaxSeqLenToTheTokenInputBound) {
  auto module = load_fixture("ET_PREFILL_CHUNK_BOUNDED_PATH");
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(get_max_prefill_chunk_size(module.get(), "forward", 8), 7);
}

TEST(PrefillChunkSizeTest, KeepsMaxSeqLenWhenTheInputAcceptsIt) {
  auto module = load_fixture("ET_PREFILL_CHUNK_FULL_PATH");
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(get_max_prefill_chunk_size(module.get(), "forward", 8), 8);
}

TEST(PrefillChunkSizeTest, NeverRaisesTheChunkAboveMaxSeqLen) {
  auto module = load_fixture("ET_PREFILL_CHUNK_FULL_PATH");
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(get_max_prefill_chunk_size(module.get(), "forward", 4), 4);
}

TEST(PrefillChunkSizeTest, FallsBackToMaxSeqLenWithoutTheMethod) {
  auto module = load_fixture("ET_PREFILL_CHUNK_BOUNDED_PATH");
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(get_max_prefill_chunk_size(module.get(), "no_such_method", 8), 8);
}

} // namespace
