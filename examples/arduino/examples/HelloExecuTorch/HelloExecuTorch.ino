/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// HelloExecuTorch — Minimal ExecuTorch sketch
//
// Initializes the ExecuTorch runtime and loads a model using the core
// ET library (portable ops only, no hardware-specific backends).
// Use this to verify the library works on your board.

#include <ExecuTorchArduino.h>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Generate it with export_model.py (see README)."
#endif

using executorch::extension::BufferDataLoader;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Program;
using executorch::runtime::Result;


void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("=== HelloExecuTorch ===");

  executorch::runtime::runtime_init();
  Serial.println("Runtime initialized.");

  auto loader = BufferDataLoader(model_pte, sizeof(model_pte));
  Result<Program> program = Program::load(&loader);
  if (program.ok()) {
    Serial.println("Model loaded OK!");
    Serial.print("  Size: ");
    Serial.print(sizeof(model_pte));
    Serial.println(" bytes");
    Serial.print("  Methods: ");
    Serial.println(program->num_methods());
  } else {
    Serial.println("ERROR: Model load failed");
  }
}

void loop() {
  Serial.println("ExecuTorch ready");
  delay(5000);
}
