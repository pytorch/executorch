/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AddModel — Verified end-to-end ExecuTorch inference on Arduino
//
// Runs a simple add model (x + 1.0) using portable ops.
// Input: [1.0, 2.0, 3.0] → Output: [2.0, 3.0, 4.0]
//
// This example uses the native ExecuTorch C++ API with no backend-specific
// ops — it works on any Arduino board that supports the ExecuTorch library.
//
// To generate model.h:
//   1. Export the model:  python -c "
//        import torch; from executorch.exir import to_edge; from torch.export import export
//        class Add(torch.nn.Module):
//            def forward(self, x): return x + 1.0
//        et = to_edge(export(Add().eval(), (torch.tensor([1.,2.,3.]),))).to_executorch()
//        with open('add.pte','wb') as f: f.write(bytes(et.buffer))"
//   2. Convert to header: python examples/arm/executor_runner/pte_to_header.py \
//        -p add.pte -o model.h

#include <ExecuTorchArduino.h>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Export a .pte and convert with pte_to_header.py (see comment above)."
#endif
#include <utility>

using executorch::aten::ScalarType;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

alignas(16) static uint8_t method_pool[8 * 1024];

static BufferDataLoader* g_loader = nullptr;
static Program* g_prog = nullptr;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== ExecuTorch Add Model ===");

  executorch::runtime::runtime_init();

  static BufferDataLoader loader(model_pte, sizeof(model_pte));
  g_loader = &loader;
  Result<Program> program = Program::load(g_loader);
  if (!program.ok()) {
    Serial.print("FAIL: load 0x");
    Serial.println((int)program.error(), HEX);
    while (1) delay(1000);
  }
  Serial.print("Model: ");
  Serial.print(sizeof(model_pte));
  Serial.println(" bytes");

  static Program prog = std::move(program.get());
  g_prog = &prog;
  Serial.println("Ready!");
}

void loop() {
  auto name_result = g_prog->get_method_name(0);
  if (!name_result.ok()) {
    Serial.println("FAIL: no methods");
    delay(5000);
    return;
  }
  const char* name = *name_result;
  auto meta = g_prog->method_meta(name);
  if (!meta.ok()) {
    Serial.println("FAIL: method_meta");
    delay(5000);
    return;
  }

  MemoryAllocator ma(sizeof(method_pool), method_pool);
  size_t np = meta->num_memory_planned_buffers();
  Span<uint8_t>* sp = static_cast<Span<uint8_t>*>(
      ma.allocate(np * sizeof(Span<uint8_t>)));
  if (!sp) { Serial.println("FAIL: OOM spans"); delay(5000); return; }
  for (size_t i = 0; i < np; i++) {
    auto sz_r = meta->memory_planned_buffer_size(i);
    if (!sz_r.ok()) { Serial.println("FAIL: planned buf size"); delay(5000); return; }
    size_t sz = static_cast<size_t>(sz_r.get());
    uint8_t* buf = static_cast<uint8_t*>(ma.allocate(sz));
    if (!buf) { Serial.println("FAIL: OOM planned buf"); delay(5000); return; }
    sp[i] = {buf, sz};
  }
  HierarchicalAllocator pl({sp, np});
  MemoryManager mm(&ma, &pl);

  auto method = g_prog->load_method(name, &mm);
  if (!method.ok()) {
    Serial.print("FAIL: load_method 0x");
    Serial.println((int)method.error(), HEX);
    delay(5000);
    return;
  }

  float input_data[] = {1.0f, 2.0f, 3.0f};
  int32_t sizes[] = {3};
  uint8_t dim_order[] = {0};
  executorch::aten::TensorImpl impl(
      ScalarType::Float, 1, sizes, input_data, dim_order);
  executorch::aten::Tensor input(&impl);
  Error err = method->set_input(EValue(input), 0);
  if (err != Error::Ok) {
    Serial.print("FAIL: set_input 0x");
    Serial.println((int)err, HEX);
    delay(5000);
    return;
  }

  Error status = method->execute();
  if (status != Error::Ok) {
    Serial.print("FAIL: execute 0x");
    Serial.println((int)status, HEX);
    delay(5000);
    return;
  }

  EValue output;
  err = method->get_outputs(&output, 1);
  if (err != Error::Ok) {
    Serial.print("FAIL: get_outputs 0x");
    Serial.println((int)err, HEX);
    delay(5000);
    return;
  }
  if (output.isTensor()) {
    auto tensor = output.toTensor();
    Serial.print("[1,2,3] + 1 = [");
    for (int i = 0; i < tensor.numel(); i++) {
      if (i > 0) Serial.print(", ");
      Serial.print(tensor.const_data_ptr<float>()[i]);
    }
    Serial.println("]");
  }

  delay(3000);
}
