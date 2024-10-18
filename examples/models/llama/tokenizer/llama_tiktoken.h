/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/tokenizer/tiktoken.h>

namespace example {

enum class Version {
  Default,
  Multimodal,
};

std::unique_ptr<::executorch::extension::llm::Tiktoken> get_tiktoken_for_llama(
    Version version = Version::Default);

} // namespace example
