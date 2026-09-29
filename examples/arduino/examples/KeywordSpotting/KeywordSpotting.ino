/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// KeywordSpotting — int8 DS-CNN through CMSIS-NN on a Cortex-M
//
// Classifies one second of speech into twelve classes: ten keywords plus
// silence and unknown. The input is a precomputed MFCC of a real recording
// from Google Speech Commands, so the sketch needs no microphone.
//
// To regenerate model.h, see the keyword spotting section of ../../README.md.

#include <ETModel.h>
#include <cstring>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Generate it with export_model.py (see README)."
#endif

// *** Change this line to test different keywords ***
#include "mfcc_yes.h"

static const char* kLabels[] = {"silence", "unknown", "yes",  "no",
                                "up",      "down",    "left", "right",
                                "on",      "off",     "stop", "go"};

// 28 KB fails load_method with MemoryAllocationFailed (0x21), and going the
// other way is not free: Zephyr takes a 32 KB stack and a 32 KB heap out of the
// board's RAM before the sketch sees any, so an arena large enough to push
// globals past that reservation overruns it. arduino-cli's RAM figure does not
// account for it and will look comfortable either way. See "Memory" in
// ../../README.md.
alignas(16) static uint8_t method_pool[40 * 1024];

static ETModel model(model_pte, sizeof(model_pte), method_pool,
                     sizeof(method_pool));

// ExecuTorch's own diagnostics arrive here. Without this they are discarded
// and failures show up as bare status codes.
extern "C" void et_arduino_log(const char* msg) {
  Serial.print("ET| ");
  Serial.println(msg);
}

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("=== ExecuTorch Keyword Spotting ===");
  Serial.print("Testing: ");
  Serial.println(test_label);

  if (!model.begin()) {
    Serial.println(model.error());
    while (1) delay(1000);
  }

  if (!model.setInput(0, test_input, 490) || !model.run()) {
    Serial.println(model.error());
    while (1) delay(1000);
  }

  const float* scores = model.output();
  for (size_t i = 0; i < model.outputCount(); i++) {
    Serial.print("  [");
    Serial.print(kLabels[i]);
    Serial.print("]=");
    Serial.println(scores[i]);
  }

  const char* detected = kLabels[model.argmax()];
  Serial.print("\n>>> Detected: ");
  Serial.println(detected);
  if (strcmp(test_label, detected) == 0) {
    Serial.println(">>> CORRECT!");
  } else {
    Serial.print(">>> Expected: ");
    Serial.println(test_label);
  }

  Serial.println("=== DONE ===");
}

void loop() {
  delay(10000);
}
