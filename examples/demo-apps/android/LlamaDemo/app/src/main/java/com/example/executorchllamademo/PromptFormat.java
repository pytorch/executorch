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
      case LLAMA_3_2:
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
      case LLAMA_3_2:
      case LLAMA_GUARD_3:
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
      case LLAMA_3_2:
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
      case LLAMA_3_2:
      case LLAMA_GUARD_3:
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

  public static String getFormattedLlamaGuardPrompt(String userPrompt) {
    return getUserPromptTemplate(ModelType.LLAMA_GUARD_3)
        .replace(
            USER_PLACEHOLDER, getLlamaGuardPresetPrompt().replace(USER_PLACEHOLDER, userPrompt));
  }

  public static String getLlamaGuardPresetPrompt() {
    return "Task: Check if there is unsafe content in 'User' messages in conversations"
        + " according our safety policy with the below categories.\n"
        + "\n"
        + "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        + "S1: Violent Crimes. \n"
        + "S2: Non-Violent Crimes. \n"
        + "S3: Sex Crimes. \n"
        + "S4: Child Exploitation. \n"
        + "S5: Defamation. \n"
        + "S6: Specialized Advice. \n"
        + "S7: Privacy. \n"
        + "S8: Intellectual Property. \n"
        + "S9: Indiscriminate Weapons. \n"
        + "S10: Hate. \n"
        + "S11: Self-Harm. \n"
        + "S12: Sexual Content. \n"
        + "S13: Elections. \n"
        + "S14: Code Interpreter Abuse.\n"
        + "<END UNSAFE CONTENT CATEGORIES>\n"
        + "\n"
        + "<BEGIN CONVERSATION>\n"
        + "\n"
        + "User: "
        + USER_PLACEHOLDER
        + "\n"
        + "<END CONVERSATION>\n"
        + "Provide your safety assessment for ONLY THE LAST User message in the above"
        + " conversation:\n"
        + " - First line must read 'safe' or 'unsafe'.\n"
        + " - If unsafe, a second line must include a comma-separated list of violated"
        + " categories.";
  }
}
