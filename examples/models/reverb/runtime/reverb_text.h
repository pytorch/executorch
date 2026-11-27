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

namespace reverb {

// Languages the released model speaks.
bool is_supported_language(const std::string& language);

// The backbone's prompt: the reference transcript, then the text to speak.
std::string build_prompt(
    const std::string& text,
    const std::string& reference_text,
    const std::string& language);

// The spoken weight of UTF-8 text: letters count 1, digits 3.5, punctuation
// and symbols 0.5, spaces 0.2 and combining marks 0.
double spoken_weight(const std::string& text);

// The target length in codec frames: the text's spoken weight at the rate the
// reference speaks its transcript. Short texts get relatively more time.
int estimate_frames(
    const std::string& text,
    const std::string& reference_text,
    int reference_frames);

// The text whole when it is estimated at most `longest` frames, else cut at
// sentence ends into pieces of about `piece` frames. A sentence longer than a
// piece stays whole.
std::vector<std::string> split_text(
    const std::string& text,
    const std::string& reference_text,
    int reference_frames,
    int longest,
    int piece);

} // namespace reverb
