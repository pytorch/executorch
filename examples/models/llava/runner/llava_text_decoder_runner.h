/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in Llava and return the output.

#pragma once

#include <executorch/extension/llm/runner/text_decoder_runner.h>

namespace torch::executor {

class LlavaTextDecoderRunner : public TextDecoderRunner {
 public:
  LlavaTextDecoderRunner(Module* module, int32_t vocab_size, float temperature)
      : TextDecoderRunner(module, true, vocab_size, temperature){};

  Result<exec_aten::Tensor> step(
      ManagedTensor& managed_tokens,
      ManagedTensor& managed_start_pos) {
    auto tokens = managed_tokens.get_aliasing_tensor();
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    // run token embedding
    std::vector<EValue> token_embedding_outputs =
        ET_UNWRAP(module_->execute("token_embedding", {tokens}));

    // run text model
    std::vector<EValue> outputs_res = ET_UNWRAP(module_->execute(
        "text_decoder", {start_pos, token_embedding_outputs[0]}));

    ET_CHECK_MSG(
        outputs_res.size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res[0].toTensor();
  }
};

} // namespace torch::executor