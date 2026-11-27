/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "parakeet_helper_protocol.h"
#include "parakeet_transcriber.h"

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");

namespace {

constexpr int kExpectedSampleRate = 16000;
constexpr int kExpectedChannelCount = 1;
constexpr const char* kExpectedEncoding = "f32le";

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  try {
    parakeet::ParakeetTranscriber transcriber(
        FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_data_path);
    if (!parakeet::helper_protocol::write_message(
            std::cout, parakeet::helper_protocol::encode_ready_message())) {
      std::cerr << "Failed to write helper ready message." << std::endl;
      return 1;
    }

    while (true) {
      parakeet::helper_protocol::Request request;
      std::string request_error;
      if (!parakeet::helper_protocol::read_request(
              std::cin, &request, &request_error)) {
        if (request_error.empty()) {
          return 0;
        }
        parakeet::helper_protocol::write_message(
            std::cout,
            parakeet::helper_protocol::encode_error_message(
                std::nullopt, "Failed to read helper request", request_error));
        return 1;
      }

      if (request.type == parakeet::helper_protocol::Request::Type::Shutdown) {
        return 0;
      }

      const auto& transcribe_request = *request.transcribe;
      try {
        if (transcribe_request.audio.encoding != kExpectedEncoding) {
          throw std::runtime_error("Unsupported audio encoding.");
        }
        if (transcribe_request.audio.sample_rate != kExpectedSampleRate) {
          throw std::runtime_error("Unsupported audio sample rate.");
        }
        if (transcribe_request.audio.channel_count != kExpectedChannelCount) {
          throw std::runtime_error("Unsupported audio channel count.");
        }
        if (transcribe_request.audio.payload_byte_count % sizeof(float) != 0) {
          throw std::runtime_error("Audio payload must be float32-aligned.");
        }

        std::string payload_bytes;
        std::string payload_error;
        if (!parakeet::helper_protocol::read_audio_payload(
                std::cin,
                transcribe_request.audio.payload_byte_count,
                &payload_bytes,
                &payload_error)) {
          throw std::runtime_error(payload_error);
        }

        std::vector<float> audio(
            transcribe_request.audio.payload_byte_count / sizeof(float));
        std::memcpy(
            audio.data(),
            payload_bytes.data(),
            transcribe_request.audio.payload_byte_count);

        const auto result = transcriber.transcribe_audio(
            audio.data(),
            static_cast<int64_t>(audio.size()),
            parakeet::TranscribeConfig{
                parakeet::parse_timestamp_output_mode("none"),
                transcribe_request.enable_runtime_profile,
            },
            [&](const std::string& status) {
              std::string phase = "status";
              if (status == "Loading recording...") {
                phase = "loading_recording";
              } else if (status == "Running preprocessor...") {
                phase = "running_preprocessor";
              } else if (status == "Running encoder...") {
                phase = "running_encoder";
              } else if (status == "Decoding final transcript...") {
                phase = "decoding_final_transcript";
              } else if (status == "Computing timestamps...") {
                phase = "computing_timestamps";
              }
              parakeet::helper_protocol::write_message(
                  std::cout,
                  parakeet::helper_protocol::encode_status_message(
                      transcribe_request.request_id, phase, status));
            });

        const std::string stdout_payload = result.stats_json.empty()
            ? std::string()
            : "PyTorchObserver " + result.stats_json;
        const auto runtime_profile_line =
            parakeet::extract_runtime_profile_line(
                result.runtime_profile_report);
        parakeet::helper_protocol::write_message(
            std::cout,
            parakeet::helper_protocol::encode_result_message(
                transcribe_request.request_id,
                result.text,
                stdout_payload,
                "",
                runtime_profile_line));
      } catch (const std::exception& e) {
        parakeet::helper_protocol::write_message(
            std::cout,
            parakeet::helper_protocol::encode_error_message(
                transcribe_request.request_id,
                "Helper transcription failed",
                e.what()));
      }
    }
  } catch (const std::exception& e) {
    parakeet::helper_protocol::write_message(
        std::cout,
        parakeet::helper_protocol::encode_error_message(
            std::nullopt, "Failed to start Parakeet helper", e.what()));
    return 1;
  }
}
