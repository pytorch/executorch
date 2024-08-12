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

namespace torch::executor {

// Assuming kv cache and parallel prefill are enabled.
class ImagePrefiller {
 public:
  ImagePrefiller(Module* module) : module_(module){};
  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to the multimodal LLM.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * @return The next token of the LLM Module after prefill.
   */
  virtual Result<exec_aten::Tensor> prefill(
      Image& image,
      int64_t start_pos = 0) = 0;

 protected:
  Module* module_;
};

} // namespace torch::executor