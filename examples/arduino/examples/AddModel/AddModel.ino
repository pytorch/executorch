/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AddModel — end-to-end ExecuTorch inference on Arduino
//
// Runs x + 1.0 with portable ops: [1, 2, 3] -> [2, 3, 4]. No backend-specific
// operators, so this works on any board the library supports.
//
// To regenerate model.h:
//   1. Export:  python -c "
//        import torch; from executorch.exir import to_edge; from torch.export import export
//        class Add(torch.nn.Module):
//            def forward(self, x): return x + 1.0
//        et = to_edge(export(Add().eval(), (torch.tensor([1.,2.,3.]),))).to_executorch()
//        with open('add.pte','wb') as f: f.write(bytes(et.buffer))"
//   2. Convert:  python examples/arduino/pte_to_header.py -p add.pte -o model.h

#include <ETModel.h>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Export a .pte and convert with pte_to_header.py (see comment above)."
#endif

// Holds the method and every tensor it works on. Too small and begin() fails
// saying so.
alignas(16) static uint8_t method_pool[8 * 1024];

static ETModel model(model_pte, sizeof(model_pte), method_pool,
                     sizeof(method_pool));

static const float kInput[] = {1.0f, 2.0f, 3.0f};

// ExecuTorch's own diagnostics arrive here. Without this they are discarded
// and failures show up as bare status codes.
extern "C" void et_arduino_log(const char* msg) {
  Serial.print("ET| ");
  Serial.println(msg);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== ExecuTorch Add Model ===");

  if (!model.begin()) {
    Serial.println(model.error());
    while (1) delay(1000);
  }
  Serial.print("Model: ");
  Serial.print(sizeof(model_pte));
  Serial.println(" bytes");
  Serial.println("Ready!");
}

void loop() {
  if (!model.setInput(0, kInput, 3) || !model.run()) {
    Serial.println(model.error());
    delay(5000);
    return;
  }

  const float* out = model.output();
  Serial.print("[1,2,3] + 1 = [");
  for (size_t i = 0; i < model.outputCount(); i++) {
    if (i > 0) Serial.print(", ");
    Serial.print(out[i]);
  }
  Serial.println("]");

  delay(3000);
}
