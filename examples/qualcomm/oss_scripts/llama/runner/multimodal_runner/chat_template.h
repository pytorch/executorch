/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/vision_chat_template.h>
#include <executorch/runtime/platform/log.h>
#include <string>
#include <vector>

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
    const std::vector<std::string>& image_paths) {
  size_t num_images = image_paths.size();
  size_t total_image_tokens = 0;

  // Count total image tokens across all prompts
  for (const auto& prompt : prompts) {
    size_t pos = 0;
    while ((pos = prompt.find(IMG_TOKEN, pos)) != std::string::npos) {
      total_image_tokens++;
      pos += IMG_TOKEN.length();
    }
  }

  // If no image tokens but images provided, prepend image tokens to prompt in
  // first turn and check the number of image tokens given by user are equal to
  // image num.
  if (total_image_tokens == 0 && num_images > 0) {
    std::string prefix;
    for (size_t i = 0; i < num_images; ++i) {
      prefix += IMG_TOKEN;
    }
    prompts[0] = prefix + prompts[0];
  }
  ET_CHECK_MSG(
      total_image_tokens == num_images,
      "Number of %s tokens (%zu) does not match number of images (%zu). Please check your prompts and image paths.",
      IMG_TOKEN.c_str(),
      total_image_tokens,
      num_images);

  // Build messages and dispatch images
  std::vector<Message> messages;
  size_t img_idx = 0;
  ET_LOG(Info, "Simulation multi-turn:");

  for (size_t i = 0; i < prompts.size(); ++i) {
    Message msg;
    msg.id = i;
    msg.text = prompts[i];

    // Count image tokens in this prompt
    size_t count = 0;
    size_t pos = 0;
    while ((pos = msg.text.find(IMG_TOKEN, pos)) != std::string::npos) {
      count++;
      pos += IMG_TOKEN.length();
    }

    // Assign corresponding images to this message
    if (count > 0) {
      for (size_t k = 0; k < count && img_idx < image_paths.size(); ++k) {
        msg.files_path.emplace_back(image_paths[img_idx++]);
      }
    }

    // Log message info
    std::string paths_str = "[";
    for (size_t i = 0; i < msg.files_path.size(); ++i) {
      paths_str += "'";
      paths_str += msg.files_path[i];
      paths_str += "'";
      if (i < msg.files_path.size() - 1)
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
