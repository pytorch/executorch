/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of LLaVA.

#pragma once

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch::executor {

class LlavaImagePrefiller : public ImagePrefiller {
 public:
  LlavaImagePrefiller(Module* module) : ImagePrefiller(module){};
  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to LLaVa.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * @return logits of the image prefill.
   */
  inline Result<exec_aten::Tensor> prefill(
      Image& image,
      int64_t start_pos = 0) {
    ManagedTensor managed_images(
        image.data.data(), {3, image.height, image.width}, ScalarType::Byte);
    // Run image encoder
    std::vector<EValue> image_encoder_outputs = ET_UNWRAP(module_->execute(
        "image_encoder", {managed_images.get_aliasing_tensor()}));

    // inputs:[start_pos, embeds]
    ManagedTensor managed_start_pos(&start_pos, {1}, ScalarType::Long);
    auto start_pos_tensor = managed_start_pos.get_aliasing_tensor();

    // Run text model
    std::vector<EValue> outputs_res = ET_UNWRAP(module_->execute(
        "text_decoder", {start_pos_tensor, image_encoder_outputs[0]}));
    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing image prefill");

    return outputs_res[0].toTensor();
  }
};

} // namespace torch::executor