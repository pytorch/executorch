/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/runtime/platform/log.h>
#include <string>
#include <vector>

const std::string IMG_TOKEN = "<image>";

/**
 * Special tokens structure for vision modality
 */
struct SpecialTokens {
  std::string image_token;
  std::string global_img;
  std::string fake_wrap_start;
  std::string fake_wrap_end;
};

/**
 * Get special tokens based on model version
 */
inline SpecialTokens get_special_tokens(
    example::VisionLanguageModel model_version) {
  SpecialTokens tokens;

  switch (model_version) {
    case example::VisionLanguageModel::kSmolvlm:
      tokens.image_token = "<image>";
      tokens.global_img = "<global-img>";
      tokens.fake_wrap_start = "<fake_token_around_image>";
      tokens.fake_wrap_end = "<fake_token_around_image>";
      break;
    case example::VisionLanguageModel::kInternvl3:
      tokens.image_token = "<IMG_CONTEXT>";
      tokens.global_img = "";
      tokens.fake_wrap_start = "<img>";
      tokens.fake_wrap_end = "</img>";
      break;
    default:
      break;
  }

  return tokens;
}

/**
 * Expand image tokens in prompt with model-specific wrapping tokens
 * Replaces each <image> token with the full format including special wrapper
 * tokens
 */
inline std::string expand_image_tokens(
    const std::string& prompt,
    const SpecialTokens& specials) {
  // Create image prompt with repeated image tokens
  std::string image_prompt = specials.fake_wrap_start;
  image_prompt += specials.global_img;
  image_prompt += specials.image_token;
  image_prompt += specials.fake_wrap_end;

  // Replace single image token with expanded version
  size_t pos = 0;
  std::string expanded = prompt;
  while ((pos = expanded.find(IMG_TOKEN, pos)) != std::string::npos) {
    expanded.replace(pos, IMG_TOKEN.size(), image_prompt);
    pos += image_prompt.size();
  }
  ET_LOG(Info, "Prompt after expanding image token: %s", expanded.c_str());

  return expanded;
}

/**
 * Format prompt based on model version with multimodal token expansion
 */
inline std::string apply_chat_template(
    const std::string& system_prompt,
    const std::string& prompt,
    example::VisionLanguageModel model_version) {
  std::string formatted_prompt;
  SpecialTokens specials = get_special_tokens(model_version);

  switch (model_version) {
    case example::VisionLanguageModel::kSmolvlm: {
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|im_start|>User:");
      formatted_prompt.append(expand_image_tokens(prompt, specials));
      formatted_prompt.append("<end_of_utterance>\nAssistant:");
      break;
    }
    case example::VisionLanguageModel::kInternvl3: {
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system<|im_end|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>");
      }
      formatted_prompt.append("<|im_start|>user:\n");
      formatted_prompt.append(expand_image_tokens(prompt, specials));
      formatted_prompt.append("<|im_end|>assistant\n");
      break;
    }
    default:
      ET_CHECK_MSG(false, "unsupported VLM version");
      break;
  }
  return formatted_prompt;
}
