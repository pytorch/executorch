/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

namespace supertonic {

struct VoiceStyle {
  std::vector<float> ttl;
  std::vector<float> dp;
};

std::string require_single_voice_style_path(
    const std::vector<std::string>& style_paths);
VoiceStyle load_voice_style(const std::string& style_path);

} // namespace supertonic
