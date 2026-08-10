/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Muse Glimmer worker for the shared LLM server. Image requests add one
// attachment segment before normal generation.

#include <gflags/gflags.h>

#include <executorch/examples/llm_server/cpp/worker_loop.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_session.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "Hugging Face tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for delegated weights.");
DEFINE_string(
    pos_embed_path,
    "",
    "Optional pos_embed.bin path; defaults beside model.pte.");
DEFINE_int64(
    max_image_bytes,
    20 * 1024 * 1024,
    "Maximum decoded JPEG/PNG byte count accepted from the control plane.");
DEFINE_int32(
    max_sessions,
    1,
    "Max physical sessions to host on one weight allocation. Clamped to 1 if "
    "the backend cannot isolate per-session mutable state.");
DEFINE_bool(
    warm_resume,
    true,
    "Warm append-only resume for named sessions when the engine supports them.");
DEFINE_int32(
    bos_id,
    200000,
    "BOS token id to prepend to server-rendered prompts.");
DEFINE_int32(eos_id, 200001, "EOS token id (MuseGlimmer convention: 200001).");
DEFINE_string(
    artifact_mode,
    "auto",
    "Artifact execution mode: auto, autoregressive, or dflash.");
DEFINE_int32(
    dflash_block_length,
    4,
    "DFlash block length. Zero uses the artifact's exported maximum.");
DEFINE_int32(
    dflash_n_draft,
    3,
    "DFlash candidates per iteration. Zero uses block_length - 1. On CUDA this "
    "must be at most 3.");
DEFINE_bool(
    dflash_draft_argmax,
    true,
    "Use exact deterministic argmax proposals for stochastic DFlash drafting.");
DEFINE_bool(
    cuda_graph,
    false,
    "Capture CUDA graphs for supported methods. Disables CUDA multi-session "
    "state rebinding.");

namespace {
namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
using Json = nlohmann::json;

llm::MuseGlimmerArtifactMode parse_artifact_mode(const std::string& value) {
  if (value == "auto") {
    return llm::MuseGlimmerArtifactMode::Auto;
  }
  if (value == "autoregressive") {
    return llm::MuseGlimmerArtifactMode::Autoregressive;
  }
  if (value == "dflash") {
    return llm::MuseGlimmerArtifactMode::DFlash;
  }
  ET_LOG(
      Error,
      "muse_glimmer_worker: invalid --artifact_mode=%s; expected auto, "
      "autoregressive, or dflash",
      value.c_str());
  return llm::MuseGlimmerArtifactMode::Auto;
}

struct ImageAttachment {
  std::string base64_data;
  size_t segment_index;
};

std::optional<ImageAttachment> extract_image_attachment(Json& request) {
  if (!request.contains("prompt_segments")) {
    return std::nullopt;
  }
  std::optional<ImageAttachment> attachment;
  size_t segment_index = 0;
  for (const auto& segment : request.at("prompt_segments")) {
    if (!segment.contains("image")) {
      ++segment_index;
      continue;
    }
    if (attachment.has_value() || segment.size() != 1 ||
        !segment.at("image").is_object()) {
      throw std::runtime_error("exactly one valid image segment is supported");
    }
    const auto& image = segment.at("image");
    if (image.value("encoding", std::string{}) != "base64") {
      throw std::runtime_error("image segment encoding must be base64");
    }
    const std::string mime_type = image.value("mime_type", std::string{});
    if (mime_type != "image/jpeg" && mime_type != "image/png") {
      throw std::runtime_error("image segment must be JPEG or PNG");
    }
    attachment =
        ImageAttachment{image.at("data").get<std::string>(), segment_index};
    ++segment_index;
  }
  return attachment;
}

void replace_image_segment_with_ids(
    Json& request,
    const ImageAttachment& attachment,
    int64_t num_soft_tokens) {
  const std::vector<uint64_t> image_ids(
      static_cast<size_t>(num_soft_tokens), llm::kMuseGlimmerImagePatchTokenId);
  request["prompt_segments"][attachment.segment_index] = {{"ids", image_ids}};
}

void handle_session_operation(
    llm::WorkerSessions& sessions,
    const Json& request) {
  const std::string op = request.value("op", std::string{});
  const std::string id = request.at("session_id").get<std::string>();
  if (id.empty()) {
    throw std::runtime_error("session_id required for op");
  }
  if (op == "close") {
    sessions.close_named(id);
    llm::worker_emit({{"closed", true}, {"session_id", id}});
    return;
  }
  if (op == "reset") {
    if (sessions.reset_named(id) != Error::Ok) {
      llm::worker_emit({{"error", "session reset failed"}, {"session_id", id}});
    } else {
      llm::worker_emit({{"reset", true}, {"session_id", id}});
    }
    return;
  }
  std::string code;
  if (sessions.open_named(id, code) == nullptr) {
    llm::worker_emit(
        {{"error", "cannot open session"}, {"code", code}, {"session_id", id}});
  } else {
    llm::worker_emit({{"opened", true}, {"session_id", id}});
  }
}

int run_muse_glimmer_worker_loop(llm::MuseGlimmerEngine& engine) {
  llm::WorkerSessions sessions(engine);
  llm::worker_emit(
      {{"ready", true},
       {"max_sessions",
        engine.serving_capacity()
            .max_physical_sessions_without_weight_duplication},
       {"max_named_sessions", sessions.max_named()},
       {"supports_image", engine.has_vision()}});

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      Json request = Json::parse(line);
      const std::string op = request.value("op", std::string{});
      if (op == "open" || op == "close" || op == "reset") {
        handle_session_operation(sessions, request);
        continue;
      }

      const std::string id = request.value("session_id", std::string{});
      llm::WorkerSessionState* state = nullptr;
      bool warm = false;
      if (id.empty()) {
        state = sessions.scratch();
      } else {
        std::string code;
        state = sessions.open_named(id, code);
        if (state == nullptr) {
          llm::worker_emit(
              {{"error", "cannot open session"},
               {"code", code},
               {"session_id", id}});
          continue;
        }
        warm = FLAGS_warm_resume;
      }

      std::vector<uint64_t> prefix = {static_cast<uint64_t>(FLAGS_bos_id)};
      Json additional_terminal_stats = Json::object();
      auto attachment = extract_image_attachment(request);
      llm::MuseGlimmerMultimodalSession* multimodal = nullptr;
      if (attachment.has_value()) {
        if (!engine.has_vision()) {
          throw std::runtime_error(
              "Muse Glimmer artifact does not support image input");
        }
        multimodal =
            llm::as_muse_glimmer_multimodal_session(state->session.get());
        if (multimodal == nullptr) {
          throw std::runtime_error(
              "image input requires a Muse Glimmer session");
        }
        auto decoded = llm::decode_muse_glimmer_base64_strict(
            attachment->base64_data,
            static_cast<size_t>(FLAGS_max_image_bytes));
        if (decoded.error() != Error::Ok) {
          throw std::runtime_error("invalid base64 image payload");
        }
        auto prepared = engine.prepare_image_from_bytes(decoded.get());
        if (prepared.error() != Error::Ok) {
          throw std::runtime_error("image decode or vision encoding failed");
        }
        replace_image_segment_with_ids(
            request, *attachment, prepared->num_soft_tokens);
        additional_terminal_stats["vision_encoder_ms"] =
            prepared->vision_encoder_ms;
        multimodal->stage_image_for_next_prefill(std::move(prepared.get()));
        warm = false;
      }

      try {
        llm::worker_handle_request(
            *state,
            warm,
            *engine.tokenizer(),
            engine.metadata(),
            request,
            prefix,
            additional_terminal_stats);
      } catch (...) {
        if (multimodal != nullptr) {
          multimodal->clear_staged_image();
        }
        throw;
      }
      if (multimodal != nullptr) {
        multimodal->clear_staged_image();
      }
    } catch (const std::exception& exception) {
      llm::worker_emit({{"error", std::string(exception.what())}});
    }
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(
        Error,
        "muse_glimmer_worker: --model_path and --tokenizer_path required");
    return 1;
  }
  if (FLAGS_artifact_mode != "auto" &&
      FLAGS_artifact_mode != "autoregressive" &&
      FLAGS_artifact_mode != "dflash") {
    parse_artifact_mode(FLAGS_artifact_mode);
    return 1;
  }
  if (FLAGS_max_image_bytes <= 0) {
    ET_LOG(Error, "muse_glimmer_worker: --max_image_bytes must be positive");
    return 1;
  }

  llm::MuseGlimmerConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.pos_embed_path = FLAGS_pos_embed_path;
  config.max_image_bytes = static_cast<size_t>(FLAGS_max_image_bytes);
  config.max_sessions = FLAGS_max_sessions;
  config.eos_id = FLAGS_eos_id;
  config.artifact_mode = parse_artifact_mode(FLAGS_artifact_mode);
  config.dflash_block_length = FLAGS_dflash_block_length;
  config.dflash_n_draft = FLAGS_dflash_n_draft;
  config.dflash_draft_argmax = FLAGS_dflash_draft_argmax;
  config.enable_cuda_graph = FLAGS_cuda_graph;

  auto engine_result = llm::MuseGlimmerEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "muse_glimmer_worker: failed to create engine");
    return 1;
  }
  return run_muse_glimmer_worker_loop(*engine_result.get());
}
