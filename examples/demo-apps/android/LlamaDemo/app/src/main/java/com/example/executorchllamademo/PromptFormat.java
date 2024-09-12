/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class PromptFormat {

  public static final String SYSTEM_PLACEHOLDER = "{{ system_prompt }}";
  public static final String USER_PLACEHOLDER = "{{ user_prompt }}";
  public static final String ASSISTANT_PLACEHOLDER = "{{ assistant_response }}";
  public static final String DEFAULT_SYSTEM_PROMPT = "Answer the questions in a few sentences";

  public static String getSystemPromptTemplate(ModelType modelType) {
    switch (modelType) {
      case LLAMA_3:
      case LLAMA_3_1:
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PLACEHOLDER
            + "<|eot_id|>";
      case LLAVA_1_5:
        return "USER: ";
      default:
        return SYSTEM_PLACEHOLDER;
    }
  }

  public static String getUserPromptTemplate(ModelType modelType) {
    switch (modelType) {
      case LLAMA_3:
      case LLAMA_3_1:
        return "<|start_header_id|>user<|end_header_id|>\n"
            + USER_PLACEHOLDER
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>";

      case LLAVA_1_5:
      default:
        return USER_PLACEHOLDER;
    }
  }

  public static String getConversationFormat(ModelType modelType) {
    switch (modelType) {
      case LLAMA_3:
      case LLAMA_3_1:
        return getUserPromptTemplate(modelType) + "\n" + ASSISTANT_PLACEHOLDER + "<|eot_id|>";
      case LLAVA_1_5:
        return USER_PLACEHOLDER + " ASSISTANT:";
      default:
        return USER_PLACEHOLDER;
    }
  }

  public static String getStopToken(ModelType modelType) {
    switch (modelType) {
      case LLAMA_3:
      case LLAMA_3_1:
        return "<|eot_id|>";
      case LLAVA_1_5:
        return "</s>";
      default:
        return "";
    }
  }

  public static String getLlavaPresetPrompt() {
    return "A chat between a curious human and an artificial intelligence assistant. The assistant"
        + " gives helpful, detailed, and polite answers to the human's questions. USER: ";
  }
}
