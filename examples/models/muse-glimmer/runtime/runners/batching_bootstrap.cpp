/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/runners/batching_bootstrap.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

DEFINE_string(pte, "", "Muse Glimmer solo off-graph program");
DEFINE_string(tokenizer, "", "Matching Hugging Face tokenizer JSON");
DEFINE_string(data_path, "", "Optional external weights file");
DEFINE_string(pos_embed_path, "", "Optional vision positional table path");
DEFINE_string(backend, "mlx", "Executor backend");
DEFINE_int32(max_sessions, 4, "Maximum logical sessions");
DEFINE_int32(max_session_tokens, 0, "Context cap; zero uses model metadata");
DEFINE_int32(max_decode_sequences, 4, "Maximum decodes in a batch");
DEFINE_int32(max_inflight_requests, 4, "Maximum admitted requests");
DEFINE_int32(
    prefix_cache_entries,
    0,
    "Text/image prefix snapshots; zero disables");
DEFINE_uint64(max_image_bytes, 20 * 1024 * 1024, "Maximum encoded image bytes");
DEFINE_uint64(
    max_input_frame_bytes,
    32 * 1024 * 1024,
    "Maximum input JSONL frame bytes");
DEFINE_int32(
    max_vision_patches,
    4096,
    "Vision patch cap, further limited by artifact");
DEFINE_int64(
    max_image_pixels,
    16 * 1024 * 1024,
    "Maximum decoded image pixels");
DEFINE_uint64(bos_id, 200000, "BOS token ID");
DEFINE_uint64(eos_id, 200001, "Additional EOS token ID (never eom)");

namespace executorch::extension::llm {
std::unique_ptr<MuseGlimmerBatchingRuntime>
create_muse_glimmer_batching_runtime() {
  if (FLAGS_pte.empty() || FLAGS_tokenizer.empty() || FLAGS_max_sessions <= 0 ||
      FLAGS_max_inflight_requests <= 0 || FLAGS_max_decode_sequences <= 0 ||
      FLAGS_prefix_cache_entries < 0 || FLAGS_max_image_bytes == 0 ||
      FLAGS_max_image_bytes > 20 * 1024 * 1024 ||
      FLAGS_max_input_frame_bytes == 0 ||
      FLAGS_max_input_frame_bytes > 32 * 1024 * 1024 ||
      FLAGS_max_image_pixels <= 0 ||
      FLAGS_max_image_pixels > 16 * 1024 * 1024) {
    throw std::invalid_argument(
        "required model/tokenizer or bounded batching limits are invalid");
  }
  const int64_t physical = static_cast<int64_t>(FLAGS_max_sessions) +
      FLAGS_prefix_cache_entries + (FLAGS_prefix_cache_entries > 0 ? 1 : 0);
  if (physical > std::numeric_limits<int>::max())
    throw std::invalid_argument("too many physical sessions");
  auto result = std::make_unique<MuseGlimmerBatchingRuntime>();
  result->tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (result->tokenizer->load(FLAGS_tokenizer) != tokenizers::Error::Ok) {
    throw std::runtime_error("could not load Hugging Face tokenizer");
  }
  auto patch = result->tokenizer->encode("<|patch|>", 0, 0);
  if (!patch.ok() || patch->size() != 1 || (*patch)[0] != 200092) {
    throw std::runtime_error("tokenizer must map <|patch|> to 200092");
  }
  MuseGlimmerBackendConfig backend;
  backend.backend = FLAGS_backend;
  backend.model_path = FLAGS_pte;
  backend.data_path = FLAGS_data_path;
  backend.pos_embed_path = FLAGS_pos_embed_path;
  backend.max_sessions = static_cast<int>(physical);
  backend.max_session_tokens = FLAGS_max_session_tokens;
  backend.max_vision_patches = FLAGS_max_vision_patches;
  backend.image_limits.max_encoded_bytes = FLAGS_max_image_bytes;
  backend.image_limits.max_image_pixels = FLAGS_max_image_pixels;
  backend.bos_id = FLAGS_bos_id;
  auto created = create_muse_glimmer_backend(backend);
  if (!created.ok())
    throw std::runtime_error(
        "could not create batching Muse Glimmer executor: error " +
        std::to_string(static_cast<int>(created.error())));
  result->backend = std::move(*created);
  const size_t width = result->backend.executor->preferred_batch_tokens();
  // A width-one artifact still works: the core physically splits the batch.
  const size_t budget =
      std::max(width, static_cast<size_t>(FLAGS_max_decode_sequences) + 1);
  const size_t decodes = FLAGS_max_decode_sequences;
  auto scheduler =
      batching::DecodeFirstScheduler::create(budget, decodes, budget - decodes);
  if (!scheduler)
    throw std::runtime_error("invalid scheduler limits");
  serving::ServingRuntimeConfig serving;
  serving.max_sessions = FLAGS_max_sessions;
  serving.max_context_length = result->backend.preparation->max_context_length;
  serving.max_requests = FLAGS_max_inflight_requests;
  serving.max_pending_operations = FLAGS_max_inflight_requests;
  serving.prefix_cache_capacity = FLAGS_prefix_cache_entries;
  serving.default_stop_tokens = {FLAGS_eos_id};
  for (const char* name : {"<|eot|>", "<|end_of_text|>"}) {
    auto encoded = result->tokenizer->encode(name, 0, 0);
    if (!encoded.ok() || encoded->size() != 1)
      throw std::runtime_error("missing Muse Glimmer EOS token");
    serving.default_stop_tokens.push_back((*encoded)[0]);
  }
  auto eom = result->tokenizer->encode("<|eom|>", 0, 0);
  if (eom.ok() && eom->size() == 1) {
    serving.default_stop_tokens.erase(
        std::remove(
            serving.default_stop_tokens.begin(),
            serving.default_stop_tokens.end(),
            (*eom)[0]),
        serving.default_stop_tokens.end());
  }
  result->runtime = std::make_unique<serving::ServingRuntime>(
      *result->backend.executor,
      std::move(scheduler),
      *result->tokenizer,
      serving,
      [spec = result->backend.preparation](
          const serving::PromptPreparationContext& context,
          const serving::PromptInput& input) {
        return prepare_muse_glimmer_input(context, input, spec);
      });
  return result;
}
} // namespace executorch::extension::llm
