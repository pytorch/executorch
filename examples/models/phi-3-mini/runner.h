/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple phi-3-mini runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <memory>
#include <string>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch::executor {

class Runner {
 public:
  explicit Runner(const std::string& model_path);

  /**
   * Generates response for a given prompt.
   *
   * @param[in] prompt The prompt to generate a response for.
   * @param[in] max_seq_len The maximum length of the sequence to generate,
   * including prompt.
   */
  void generate();

  void test_example();

 private:
  std::unique_ptr<Module> module_;
};

} // namespace torch::executor
