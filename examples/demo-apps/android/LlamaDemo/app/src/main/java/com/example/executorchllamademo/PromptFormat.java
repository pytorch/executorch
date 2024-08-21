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

  public static String getSystemPromptTemplate(ModelType modelType) {
    switch (modelType) {
      case LLAMA_3:
      case LLAMA_3_1:
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PLACEHOLDER
            + "<|eot_id|>";
      case LLAVA_1_5:
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
            + "<|eot_id|>\n"
            + "<|start_header_id|>assistant<|end_header_id|>";
      case LLAVA_1_5:
      default:
        return USER_PLACEHOLDER;
    }
  }
}
