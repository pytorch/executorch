/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/audio_chat_template.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/vision_chat_template.h>
#include <executorch/runtime/platform/log.h>
#include <string>
#include <vector>

using executorch::extension::llm::MultimodalInput;

/**
 * Message structure for multi-turn conversations
 */
struct Message {
  size_t id;
  std::string text;
  std::vector<std::string> files_path;
};

/**
 * Prepare messages for multi-turn simulation
 * This function validates that the number of image tokens matches the number of
 * images, and distributes images across messages based on image token
 * positions.
 */
inline std::vector<Message> prepare_messages(
    std::vector<std::string>& prompts,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& audio_paths) {
  size_t num_images = image_paths.size();
  size_t num_audios = audio_paths.size();
  size_t total_image_tokens = 0;
  size_t total_audio_tokens = 0;

  // Count total image and audio tokens across all prompts
  for (const auto& prompt : prompts) {
    size_t pos = 0;
    while ((pos = prompt.find(IMG_TOKEN, pos)) != std::string::npos) {
      total_image_tokens++;
      pos += IMG_TOKEN.length();
    }
    pos = 0;
    while ((pos = prompt.find(AUDIO_TOKEN, pos)) != std::string::npos) {
      total_audio_tokens++;
      pos += AUDIO_TOKEN.length();
    }
  }

  auto repeat_token = [](const std::string& token, size_t n) -> std::string {
    std::string result;
    result.reserve(n * token.size());
    for (size_t i = 0; i < n; ++i)
      result += token;
    return result;
  };

  // Prepend tokens if paths are provided but tokens are missing
  if (total_image_tokens == 0 && num_images > 0) {
    prompts[0] = repeat_token(IMG_TOKEN, num_images) + prompts[0];
    total_image_tokens = num_images; // Update count
  }
  if (total_audio_tokens == 0 && num_audios > 0) {
    prompts[0] = repeat_token(AUDIO_TOKEN, num_audios) + prompts[0];
    total_audio_tokens = num_audios; // Update count
  }

  // Validate token counts against provided paths
  ET_CHECK_MSG(
      total_image_tokens == num_images,
      "Number of %s tokens (%zu) does not match number of images (%zu). Please check your prompts and image paths.",
      IMG_TOKEN.c_str(),
      total_image_tokens,
      num_images);

  ET_CHECK_MSG(
      total_audio_tokens == num_audios,
      "Number of %s tokens (%zu) does not match number of audios (%zu). Please check your prompts and audio paths.",
      AUDIO_TOKEN.c_str(),
      total_audio_tokens,
      num_audios);

  // Build messages and dispatch images/audios.
  // A model may support both vision and audio modalities simultaneously (e.g.,
  // omni models). Files are dispatched in prompt order: for each turn, we scan
  // for IMG_TOKEN and AUDIO_TOKEN tokens and assign the next available
  // image/audio path respectively, preserving interleaved ordering.
  std::vector<Message> messages;
  size_t img_idx = 0;
  size_t audio_idx = 0;
  ET_LOG(Info, "Simulation multi-turn:");
  for (size_t i = 0; i < prompts.size(); ++i) {
    Message msg;
    msg.id = i;
    msg.text = prompts[i];
    std::string& current_prompt = msg.text;

    // Collect positions of each token type separately
    std::vector<size_t> audio_positions, img_positions;
    size_t pos = 0;
    while ((pos = current_prompt.find(AUDIO_TOKEN, pos)) != std::string::npos) {
      audio_positions.push_back(pos);
      pos += AUDIO_TOKEN.length();
    }
    pos = 0;
    while ((pos = current_prompt.find(IMG_TOKEN, pos)) != std::string::npos) {
      img_positions.push_back(pos);
      pos += IMG_TOKEN.length();
    }

    // Merge into (position, Modality) and sort by position
    std::vector<std::pair<size_t, example::Modality>>
        ordered_modality_token_ids;
    ordered_modality_token_ids.reserve(
        img_positions.size() + audio_positions.size());
    for (size_t p : audio_positions) {
      ordered_modality_token_ids.emplace_back(p, example::Modality::kAudio);
    }
    for (size_t p : img_positions) {
      ordered_modality_token_ids.emplace_back(p, example::Modality::kVision);
    }
    std::sort(
        ordered_modality_token_ids.begin(), ordered_modality_token_ids.end());

    // Push file paths in order
    for (const auto& [_, modality] : ordered_modality_token_ids) {
      if (modality == example::Modality::kAudio) {
        msg.files_path.push_back(audio_paths[audio_idx++]);
      } else if (modality == example::Modality::kVision) {
        msg.files_path.push_back(image_paths[img_idx++]);
      }
    }

    // Log message info
    std::string paths_str = "[";
    for (size_t j = 0; j < msg.files_path.size(); ++j) {
      paths_str += "'";
      paths_str += msg.files_path[j];
      paths_str += "'";
      if (j < msg.files_path.size() - 1)
        paths_str += ", ";
    }
    paths_str += "]";
    ET_LOG(
        Info,
        "Turn-%zu: {id: %zu, text: \"%s\", files_path: %s}",
        i,
        i,
        msg.text.c_str(),
        paths_str.c_str());

    messages.emplace_back(std::move(msg));
  }

  return messages;
}

inline std::string apply_chat_template(
    const std::string& prompt,
    const std::string& system_prompt,
    example::ModelVersion model_version) {
  return std::visit(
      [&](const auto& model) {
        return apply_chat_template(system_prompt, prompt, model);
      },
      model_version);
}

std::vector<MultimodalInput> dispatch_inputs(
    const std::vector<MultimodalInput>& inputs,
    const std::string& formatted_prompt) {
  // Dispatch a formatted prompt into text and multimodal inputs at each
  // placeholder token position.
  //
  // VLM example (SmolVLM):
  //   inputs:  [cat.jpg (image)]
  //   prompt:  "<|im_start|>User:<fake_token_around_image><global-img><image>"
  //            "<fake_token_around_image>Can you describe this image?
  //            "<end_of_utterance>\nAssistant:"
  //   returns: ["<|im_start|>User:<fake_token_around_image><global-img>",
  //             cat.jpg,
  //             "<fake_token_around_image>Can you describe this image?
  //             "<end_of_utterance>\nAssistant:"]
  //
  // ALM example (Granite Speech):
  //   inputs:  [speech.wav (audio)]
  //   prompt:  "<|start_of_role|>user<|end_of_role|><audio>can you transcribe
  //   the speech into a written format?"
  //            "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
  //   returns: ["<|start_of_role|>user<|end_of_role|>",
  //             speech.wav,
  //             "can you transcribe the speech into a written format?
  //             "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"]
  std::vector<MultimodalInput> dispatched_inputs;
  size_t prompt_pos = 0;

  for (const auto& input : inputs) {
    if (input.is_image()) {
      size_t img_token_pos = formatted_prompt.find(IMG_TOKEN, prompt_pos);
      if (img_token_pos != std::string::npos) {
        // Add text before the image token
        if (img_token_pos > prompt_pos) {
          dispatched_inputs.emplace_back(
              formatted_prompt.substr(prompt_pos, img_token_pos - prompt_pos));
        }
        // Add the image input
        dispatched_inputs.emplace_back(input);
        // Move position over the image token
        prompt_pos = img_token_pos + IMG_TOKEN.length();
      }
    } else if (input.is_audio()) {
      size_t audio_token_pos = formatted_prompt.find(AUDIO_TOKEN, prompt_pos);
      if (audio_token_pos != std::string::npos) {
        // Add text before the audio token
        if (audio_token_pos > prompt_pos) {
          dispatched_inputs.emplace_back(formatted_prompt.substr(
              prompt_pos, audio_token_pos - prompt_pos));
        }
        // Add the audio input
        dispatched_inputs.emplace_back(input);
        // Move position over the audio token
        prompt_pos = audio_token_pos + AUDIO_TOKEN.length();
      }
    }
  }

  // Add any remaining text after the last token
  if (prompt_pos < formatted_prompt.length()) {
    dispatched_inputs.emplace_back(formatted_prompt.substr(prompt_pos));
  }

  return dispatched_inputs;
}
