/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class ModelUtils {
  static final int TEXT_MODEL = 1;
  static final int VISION_MODEL = 2;
  static final int VISION_MODEL_IMAGE_CHANNELS = 3;
  static final int VISION_MODEL_SEQ_LEN = 768;
  static final int TEXT_MODEL_SEQ_LEN = 256;

  public static int getModelCategory(ModelType modelType) {
    switch (modelType) {
      case LLAVA_1_5:
        return VISION_MODEL;
      case LLAMA_3:
      case LLAMA_3_1:
      case LLAMA_3_2:
      default:
        return TEXT_MODEL;
    }
  }
}
