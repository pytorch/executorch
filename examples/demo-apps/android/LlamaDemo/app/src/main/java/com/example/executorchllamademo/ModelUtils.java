/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class ModelUtils {
  // XNNPACK or QNN
  static final int TEXT_MODEL = 1;

  // XNNPACK
  static final int VISION_MODEL = 2;
  static final int VISION_MODEL_IMAGE_CHANNELS = 3;
  static final int VISION_MODEL_SEQ_LEN = 768;
  static final int TEXT_MODEL_SEQ_LEN = 256;

  // MediaTek
  static final int MEDIATEK_TEXT_MODEL = 3;

  public static int getModelCategory(ModelType modelType, BackendType backendType) {
    if (backendType.equals(BackendType.XNNPACK)) {
      switch (modelType) {
        case LLAVA_1_5:
          return VISION_MODEL;
        case LLAMA_3:
        case LLAMA_3_1:
        case LLAMA_3_2:
        default:
          return TEXT_MODEL;
      }
    } else if (backendType.equals(BackendType.MEDIATEK)) {
      return MEDIATEK_TEXT_MODEL;
    }

    return TEXT_MODEL; // default
  }
}
