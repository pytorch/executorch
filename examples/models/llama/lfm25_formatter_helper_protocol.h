/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LFM2.5 formatter helper: a long-lived companion process that wraps an
// `executorch::extension::llm::TextLLMRunner` with a JSON-line stdin/stdout
// protocol. The macOS ExecuWhisper app launches this binary once per app
// session, sends a `format` request, and reads the rewritten dictation off
// stdout — preserving the model's KV cache and warm state across requests.
//
// Wire contract (kProtocolVersion=1):
//
//   Requests (one JSON object per line, then optional payload):
//     {"type": "format",   "version": 1,
//      "request_id": "<uuid>",
//      "prompt": "<chat-templated prompt>",
//      "max_new_tokens": <int>,
//      "temperature": <double>}
//     {"type": "shutdown", "version": 1}
//
//   Responses (one JSON object per line):
//     {"type": "ready",   "version": 1}                      // emitted once at startup
//     {"type": "status",  "version": 1, "request_id": ...,
//      "phase": "<short>", "message": "<human>"}             // optional progress updates
//     {"type": "result",  "version": 1, "request_id": ...,
//      "text": "<generated>", "stdout": "", "stderr": "",
//      "tokens_per_second": <double?>}                       // success
//     {"type": "error",   "version": 1, "request_id": <opt>,
//      "message": "<short>", "details": <opt>}               // failure
//
// The Swift wire contract this matches lives at
//   ExecuWhisper/Services/FormatterHelperProtocol.swift
// in the internal-llama-cookbook ExecuWhisper app.

#pragma once

#include <cstddef>
#include <istream>
#include <optional>
#include <ostream>
#include <string>

namespace lfm25_formatter::helper_protocol {

constexpr int kProtocolVersion = 1;

struct FormatRequest {
  std::string request_id;
  std::string prompt;
  int max_new_tokens = 0;
  double temperature = 0.0;
};

struct Request {
  enum class Type {
    Format,
    Shutdown,
  };

  Type type = Type::Shutdown;
  std::optional<FormatRequest> format;
};

bool read_request(
    std::istream& input,
    Request* request,
    std::string* error_message);

std::string encode_ready_message();
std::string encode_status_message(
    const std::optional<std::string>& request_id,
    const std::string& phase,
    const std::string& message);
std::string encode_result_message(
    const std::string& request_id,
    const std::string& text,
    const std::string& stdout_payload,
    const std::string& stderr_payload,
    const std::optional<double>& tokens_per_second);
std::string encode_error_message(
    const std::optional<std::string>& request_id,
    const std::string& message,
    const std::optional<std::string>& details);

bool write_message(std::ostream& output, const std::string& line);

} // namespace lfm25_formatter::helper_protocol
