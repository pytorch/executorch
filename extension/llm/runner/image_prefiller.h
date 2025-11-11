/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of a multimodal LLM.

#pragma once

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

// Assuming kv cache and parallel prefill are enabled.
class ET_EXPERIMENTAL ImagePrefiller {
 public:
  explicit ImagePrefiller(::executorch::extension::Module* module)
      : module_(module) {}

  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to the multimodal LLM.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * It's passed as reference and will be updated inside this function.
   * @return The next token of the LLM Module after prefill.
   */
  virtual ::executorch::runtime::Result<executorch::aten::Tensor> prefill(
      Image& image,
      int64_t& start_pos) = 0;

  virtual ::executorch::runtime::Error load() = 0;
  virtual bool is_method_loaded() = 0;

  virtual ~ImagePrefiller() = default;

 protected:
  Module* module_;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::ImagePrefiller;
} // namespace executor
} // namespace torch
