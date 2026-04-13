/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "parakeet_helper_protocol.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <istream>
#include <optional>
#include <ostream>
#include <string>

namespace parakeet::helper_protocol {
namespace {

using json = nlohmann::json;

} // namespace

bool read_request(
    std::istream& input,
    Request* request,
    std::string* error_message) {
  std::string header_line;
  if (!std::getline(input, header_line)) {
    return false;
  }
  if (header_line.empty()) {
    if (error_message) {
      *error_message = "Received empty helper request header.";
    }
    return false;
  }

  json payload;
  try {
    payload = json::parse(header_line);
  } catch (const std::exception& e) {
    if (error_message) {
      *error_message =
          std::string("Failed to parse helper request: ") + e.what();
    }
    return false;
  }

  const std::string type = payload.value("type", "");
  if (payload.value("version", -1) != kProtocolVersion) {
    if (error_message) {
      *error_message = "Unsupported helper protocol version.";
    }
    return false;
  }

  if (type == "shutdown") {
    request->type = Request::Type::Shutdown;
    request->transcribe.reset();
    return true;
  }

  if (type != "transcribe") {
    if (error_message) {
      *error_message = "Unsupported helper request type: " + type;
    }
    return false;
  }

  if (!payload.contains("audio") || !payload["audio"].is_object()) {
    if (error_message) {
      *error_message = "Missing helper audio descriptor.";
    }
    return false;
  }

  const auto& audio = payload["audio"];
  TranscribeRequest transcribe_request;
  transcribe_request.request_id = payload.value("request_id", "");
  transcribe_request.enable_runtime_profile =
      payload.value("enable_runtime_profile", false);
  transcribe_request.audio.encoding = audio.value("encoding", "");
  transcribe_request.audio.sample_rate = audio.value("sample_rate", 0);
  transcribe_request.audio.channel_count = audio.value("channel_count", 0);
  transcribe_request.audio.payload_byte_count =
      audio.value("payload_byte_count", static_cast<std::size_t>(0));

  request->type = Request::Type::Transcribe;
  request->transcribe = transcribe_request;
  return true;
}

bool read_audio_payload(
    std::istream& input,
    std::size_t payload_byte_count,
    std::string* payload,
    std::string* error_message) {
  payload->assign(payload_byte_count, '\0');
  input.read(payload->data(), static_cast<std::streamsize>(payload_byte_count));
  if (input.gcount() != static_cast<std::streamsize>(payload_byte_count)) {
    if (error_message) {
      *error_message = "Failed to read full helper payload.";
    }
    return false;
  }
  return true;
}

std::string encode_ready_message() {
  return json{{"type", "ready"}, {"version", kProtocolVersion}}.dump();
}

std::string encode_status_message(
    const std::optional<std::string>& request_id,
    const std::string& phase,
    const std::string& message) {
  json payload = {
      {"type", "status"},
      {"version", kProtocolVersion},
      {"phase", phase},
      {"message", message},
  };
  if (request_id.has_value()) {
    payload["request_id"] = *request_id;
  }
  return payload.dump();
}

std::string encode_result_message(
    const std::string& request_id,
    const std::string& text,
    const std::string& stdout,
    const std::string& stderr,
    const std::optional<std::string>& runtime_profile) {
  json payload = {
      {"type", "result"},
      {"version", kProtocolVersion},
      {"request_id", request_id},
      {"text", text},
      {"stdout", stdout},
      {"stderr", stderr},
  };
  if (runtime_profile.has_value()) {
    payload["runtime_profile"] = *runtime_profile;
  }
  return payload.dump();
}

std::string encode_error_message(
    const std::optional<std::string>& request_id,
    const std::string& message,
    const std::optional<std::string>& details) {
  json payload = {
      {"type", "error"},
      {"version", kProtocolVersion},
      {"message", message},
  };
  if (request_id.has_value()) {
    payload["request_id"] = *request_id;
  }
  if (details.has_value()) {
    payload["details"] = *details;
  }
  return payload.dump();
}

bool write_message(std::ostream& output, const std::string& line) {
  output << line << '\n';
  output.flush();
  return output.good();
}

} // namespace parakeet::helper_protocol
