/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lfm25_formatter_helper_protocol.h"

#include <nlohmann/json.hpp>

#include <istream>
#include <optional>
#include <ostream>
#include <string>

namespace lfm25_formatter::helper_protocol {
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
    request->format.reset();
    return true;
  }

  if (type != "format") {
    if (error_message) {
      *error_message = "Unsupported helper request type: " + type;
    }
    return false;
  }

  if (!payload.contains("prompt") || !payload["prompt"].is_string()) {
    if (error_message) {
      *error_message = "Missing helper prompt field.";
    }
    return false;
  }

  FormatRequest format_request;
  format_request.request_id = payload.value("request_id", "");
  format_request.prompt = payload.value("prompt", "");
  format_request.max_new_tokens = payload.value("max_new_tokens", 0);
  format_request.temperature = payload.value("temperature", 0.0);

  request->type = Request::Type::Format;
  request->format = format_request;
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
    const std::string& stdout_payload,
    const std::string& stderr_payload,
    const std::optional<double>& tokens_per_second) {
  json payload = {
      {"type", "result"},
      {"version", kProtocolVersion},
      {"request_id", request_id},
      {"text", text},
      {"stdout", stdout_payload},
      {"stderr", stderr_payload},
  };
  if (tokens_per_second.has_value()) {
    payload["tokens_per_second"] = *tokens_per_second;
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

} // namespace lfm25_formatter::helper_protocol
