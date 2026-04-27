/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace voxtral_tts {

class WavWriter {
 public:
  WavWriter(const std::string& path, int sample_rate, int num_channels = 1);
  ~WavWriter();

  WavWriter(const WavWriter&) = delete;
  WavWriter& operator=(const WavWriter&) = delete;

  bool Write(const float* samples, std::size_t frame_count);
  bool Close();
  bool IsOpen() const;

 private:
  void WriteHeaderPlaceholder();
  bool FinalizeHeader();

  std::ofstream file_;
  int sample_rate_;
  int num_channels_;
  std::uint32_t data_bytes_ = 0;
  bool closed_ = false;
};

} // namespace voxtral_tts
