/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/models/llama2/tokenizer/tiktoken.h>

namespace torch {
namespace executor {

enum Version {
  DEFAULT,
  MULTIMODAL,
};

class LlamaTiktoken : public Tiktoken {
 public:
  explicit LlamaTiktoken(Version version = Version::DEFAULT)
      : Tiktoken(), _version(version) {}
  ~LlamaTiktoken() {}

 protected:
  const Encoder get_special_tokens(ssize_t num_base_tokens) const override;

 private:
  const Version _version;
};
} // namespace executor
} // namespace torch
