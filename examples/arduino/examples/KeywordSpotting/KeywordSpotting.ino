/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// KeywordSpotting — DS-CNN inference with ExecuTorch + CMSIS-NN
//
// Runs a quantized DS-CNN model (MLPerf Tiny KWS benchmark) on real
// MFCC audio features and prints the detected keyword.
//
// To test a different keyword, change the #include below:
//   #include "mfcc_yes.h"   → detects "yes"

//   #include "mfcc_no.h"    → detects "no"
//   #include "mfcc_stop.h"  → detects "stop"
//   etc.
//
// To test your own audio:
//   python generate_test_input.py --input my_audio.wav --output mfcc_custom.h
//   Then: #include "mfcc_custom.h"
//
// Pre-generated MFCC files from Google Speech Commands v2:
//   mfcc_yes.h, mfcc_no.h, mfcc_up.h, mfcc_down.h, mfcc_left.h,
//   mfcc_right.h, mfcc_on.h, mfcc_off.h, mfcc_stop.h, mfcc_go.h

#include <ExecuTorchArduino.h>
#include <cstring>
#include <utility>
#if __has_include("model.h")
#include "model.h"
#else
#error "model.h not found. Generate it with export_model.py (see README)."
#endif

// *** Change this line to test different keywords ***
#include "mfcc_yes.h"


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

static const char* kLabels[] = {
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"};

alignas(16) static uint8_t method_pool[28 * 1024];

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("=== ExecuTorch Keyword Spotting ===");
  Serial.print("Testing: ");
  Serial.println(test_label);

  executorch::runtime::runtime_init();

  static BufferDataLoader loader(model_pte, sizeof(model_pte));
  auto program = Program::load(&loader);
  if (!program.ok()) {
    Serial.print("FAIL: load 0x");
    Serial.println((int)program.error(), HEX);
    while (1) delay(1000);
  }

  static Program prog = std::move(program.get());
  auto name_r = prog.get_method_name(0);
  if (!name_r.ok()) { Serial.println("FAIL: name"); while (1) delay(1000); }
  auto meta = prog.method_meta(*name_r);
  if (!meta.ok()) { Serial.println("FAIL: meta"); while (1) delay(1000); }

  MemoryAllocator ma(sizeof(method_pool), method_pool);
  size_t np = meta->num_memory_planned_buffers();
  Span<uint8_t>* sp = static_cast<Span<uint8_t>*>(
      ma.allocate(np * sizeof(Span<uint8_t>)));
  if (!sp) { Serial.println("FAIL: OOM"); while (1) delay(1000); }
  for (size_t i = 0; i < np; i++) {
    auto sz_r = meta->memory_planned_buffer_size(i);
    if (!sz_r.ok()) { Serial.println("FAIL: planned buf size"); while (1) delay(1000); }
    size_t sz = static_cast<size_t>(sz_r.get());
    uint8_t* buf = static_cast<uint8_t*>(ma.allocate(sz));
    if (!buf) { Serial.println("FAIL: OOM buf"); while (1) delay(1000); }
    sp[i] = {buf, sz};
  }

  HierarchicalAllocator pl({sp, np});
  MemoryManager mm(&ma, &pl);

  auto method = prog.load_method(*name_r, &mm);
  if (!method.ok()) {
    Serial.print("FAIL: method 0x");
    Serial.println((int)method.error(), HEX);
    while (1) delay(1000);
  }

  // Set input from MFCC data
  auto imeta = meta->input_tensor_meta(0);
  if (!imeta.ok()) { Serial.println("FAIL: imeta"); while (1) delay(1000); }
  // Copy sizes/dim_order to local mutable arrays — TensorImpl stores pointers.
  const size_t ndim = imeta->sizes().size();
  if (ndim != 4 || imeta->dim_order().size() != 4) {
    Serial.println("FAIL: expected 4D input"); while (1) delay(1000);
  }
  if (imeta->scalar_type() != ScalarType::Float) {
    Serial.println("FAIL: expected float input"); while (1) delay(1000);
  }
  int32_t input_sizes[4];
  for (size_t d = 0; d < imeta->sizes().size(); d++) input_sizes[d] = imeta->sizes()[d];
  uint8_t input_dim_order[4];
  for (size_t d = 0; d < imeta->dim_order().size(); d++) input_dim_order[d] = imeta->dim_order()[d];
  static float input_data[490];
  int numel = 1;
  for (size_t d = 0; d < ndim; d++) numel *= input_sizes[d];
  if (numel != 490) {
    Serial.println("FAIL: expected 490 input elements"); while (1) delay(1000);
  }
  memcpy(input_data, test_input, sizeof(input_data));
  executorch::aten::TensorImpl iimpl(
      ScalarType::Float, imeta->sizes().size(),
      input_sizes, input_data, input_dim_order);
  executorch::aten::Tensor it(&iimpl);
  Error serr = method->set_input(EValue(it), 0);
  if (serr != Error::Ok) {
    Serial.print("FAIL: input 0x");
    Serial.println((int)serr, HEX);
    while (1) delay(1000);
  }

  // Run inference
  Error status = method->execute();
  if (status != Error::Ok) {
    Serial.print("FAIL: execute 0x");
    Serial.println((int)status, HEX);
    while (1) delay(1000);
  }

  // Read and display results
  EValue output;
  Error oerr = method->get_outputs(&output, 1);
  if (oerr != Error::Ok) {
    Serial.print("FAIL: output 0x");
    Serial.println((int)oerr, HEX);
    while (1) delay(1000);
  }

  if (output.isTensor()) {
    auto tensor = output.toTensor();
    int best = 0;
    float best_val = -1e9f;
    for (int i = 0; i < 12 && i < tensor.numel(); i++) {
      float val;
      if (tensor.scalar_type() == ScalarType::Float)
        val = tensor.const_data_ptr<float>()[i];
      else
        val = static_cast<float>(tensor.const_data_ptr<int8_t>()[i]);
      Serial.print("  [");
      Serial.print(kLabels[i]);
      Serial.print("]=");
      Serial.println(val);
      if (val > best_val) { best_val = val; best = i; }
    }

    Serial.print("\n>>> Detected: ");
    Serial.println(kLabels[best]);

    if (strcmp(test_label, kLabels[best]) == 0) {
      Serial.println(">>> CORRECT!");
    } else {
      Serial.print(">>> Expected: ");
      Serial.println(test_label);
    }
  }

  Serial.println("=== DONE ===");
}

void loop() {
  delay(10000);
}
