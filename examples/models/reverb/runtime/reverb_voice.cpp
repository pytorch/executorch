/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_voice.h"

#include "reverb_sampler.h"
#include "reverb_text.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace reverb {

int Voice::pace_frames(bool whole_recording) const {
  if (whole_recording || !speech_frames) {
    return frames();
  }
  return std::max(1, std::min(*speech_frames, frames()));
}

Voice load_voice(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open voice " + path);
  }
  const nlohmann::json data = nlohmann::json::parse(in);
  Voice voice;
  for (const auto& entry : data.at("codes")) {
    if (entry.is_array()) {
      for (const auto& code : entry) {
        voice.codes.push_back(code.get<int64_t>());
      }
    } else {
      voice.codes.push_back(entry.get<int64_t>());
    }
  }
  if (voice.codes.empty() || voice.codes.size() % kCodebooks ||
      std::any_of(voice.codes.begin(), voice.codes.end(), [](int64_t code) {
        return code < 0 || code >= kMask;
      })) {
    throw std::runtime_error(
        path + ": expected codes as 8 x frames integers below 1024");
  }
  voice.transcript = data.at("transcript").get<std::string>();
  voice.language = data.at("language").get<std::string>();
  if (!is_supported_language(voice.language)) {
    throw std::runtime_error(path + ": unsupported language " + voice.language);
  }
  if (data.contains("speech_frames") && !data["speech_frames"].is_null()) {
    const int speech = data["speech_frames"].get<int>();
    if (speech < 1) {
      throw std::runtime_error(path + ": speech_frames must be positive");
    }
    voice.speech_frames = speech;
  }
  return voice;
}

void save_voice(const Voice& voice, const std::string& path) {
  nlohmann::json data;
  data["codes"] = voice.codes;
  data["transcript"] = voice.transcript;
  data["language"] = voice.language;
  data["frames"] = voice.frames();
  if (voice.speech_frames) {
    data["speech_frames"] = *voice.speech_frames;
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write voice " + path);
  }
  out << data.dump() << "\n";
}

} // namespace reverb
