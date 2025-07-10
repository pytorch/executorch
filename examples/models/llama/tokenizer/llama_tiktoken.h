/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pytorch/tokenizers/tiktoken.h>

namespace example {

enum class Version {
  Default,
  Multimodal,
};

std::unique_ptr<::tokenizers::Tiktoken> get_tiktoken_for_llama(
    Version version = Version::Default);

std::unique_ptr<std::vector<std::string>> get_special_tokens(Version version);

std::unique_ptr<std::vector<std::string>> get_multimodal_special_tokens();

} // namespace example
