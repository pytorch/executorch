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

inline const std::string AUDIO_TOKEN = "<audio>";

struct AudioSpecialTokens {
  std::string audio_token;
};

/**
 * Get special tokens based on audio model version
 */
inline AudioSpecialTokens get_special_tokens(
    example::AudioLanguageModel model_version) {
  AudioSpecialTokens tokens;
  tokens.audio_token = AUDIO_TOKEN;

  switch (model_version) {
    case example::AudioLanguageModel::kGraniteSpeech:
      break;
    default:
      break;
  }

  return tokens;
}

/**
 * Expand audio tokens in prompt with model-specific wrapping tokens
 * Replaces each <audio> token with the full format including special wrapper
 * tokens
 */
inline std::string expand_audio_tokens(
    const std::string& prompt,
    const AudioSpecialTokens& specials) {
  // Create audio prompt with repeated audio tokens
  std::string audio_prompt = specials.audio_token;

  // Replace single audio token with expanded version
  size_t pos = 0;
  std::string expanded = prompt;
  while ((pos = expanded.find(specials.audio_token, pos)) !=
         std::string::npos) {
    expanded.replace(pos, specials.audio_token.size(), audio_prompt);
    pos += audio_prompt.size();
  }
  ET_LOG(Info, "Prompt after expanding audio token: %s", expanded.c_str());

  return expanded;
}

/**
 * Format prompt based on model version with multimodal token expansion
 */
inline std::string apply_chat_template(
    const std::string& system_prompt,
    const std::string& prompt,
    example::AudioLanguageModel model_version) {
  std::string formatted_prompt;
  AudioSpecialTokens specials = get_special_tokens(model_version);

  switch (model_version) {
    case example::AudioLanguageModel::kGraniteSpeech: {
      formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
      } else {
        formatted_prompt.append(
            "You are Granite, developed by IBM. You are a helpful AI assistant.");
      }
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(expand_audio_tokens(prompt, specials));
      formatted_prompt.append(
          "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
      break;
    }
    default:
      ET_CHECK_MSG(false, "unsupported Audio-Language model version");
      break;
  }
  return formatted_prompt;
}
