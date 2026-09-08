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

#include <ExecuTorch.h>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Generate it with export_model.py (see README)."
#endif

using executorch::extension::BufferDataLoader;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Program;
using executorch::runtime::Result;

static bool g_loaded = false;

// ExecuTorch logs go to a weak hook so the library does not depend on Serial.
// Without this the runtime's own diagnostics -- allocation failures, operator
// mismatches -- are discarded, and errors surface only as bare hex codes.
extern "C" void et_arduino_log(const char* msg) {
  Serial.print("ET| ");
  Serial.println(msg);
}

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
    g_loaded = true;
  } else {
    Serial.print("ERROR: Model load failed 0x");
    Serial.println((int)program.error(), HEX);
  }
}

void loop() {
  // Report the real state. Printing a fixed string here would look identical
  // whether or not the model loaded, and setup() has already scrolled away by
  // the time a serial monitor attaches.
  Serial.println(g_loaded ? "ExecuTorch ready" : "ExecuTorch FAILED to load");
  delay(5000);
}
