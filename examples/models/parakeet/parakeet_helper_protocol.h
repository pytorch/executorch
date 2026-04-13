/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <istream>
#include <optional>
#include <ostream>
#include <string>

namespace parakeet::helper_protocol {

constexpr int kProtocolVersion = 1;

struct AudioDescriptor {
  std::string encoding;
  int sample_rate = 0;
  int channel_count = 0;
  std::size_t payload_byte_count = 0;
};

struct TranscribeRequest {
  std::string request_id;
  AudioDescriptor audio;
  bool enable_runtime_profile = false;
};

struct Request {
  enum class Type {
    Transcribe,
    Shutdown,
  };

  Type type = Type::Shutdown;
  std::optional<TranscribeRequest> transcribe;
};

bool read_request(
    std::istream& input,
    Request* request,
    std::string* error_message);

bool read_audio_payload(
    std::istream& input,
    std::size_t payload_byte_count,
    std::string* payload,
    std::string* error_message);

std::string encode_ready_message();
std::string encode_status_message(
    const std::optional<std::string>& request_id,
    const std::string& phase,
    const std::string& message);
std::string encode_result_message(
    const std::string& request_id,
    const std::string& text,
    const std::string& stdout,
    const std::string& stderr,
    const std::optional<std::string>& runtime_profile);
std::string encode_error_message(
    const std::optional<std::string>& request_id,
    const std::string& message,
    const std::optional<std::string>& details);

bool write_message(std::ostream& output, const std::string& line);

} // namespace parakeet::helper_protocol
