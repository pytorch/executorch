/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "muse_glimmer_batching_workload.h"

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

DECLARE_string(pte);
DECLARE_string(tokenizer);
DECLARE_string(data_path);
DECLARE_string(backend);
DECLARE_int32(max_sessions);
DECLARE_int32(max_session_tokens);
DECLARE_uint64(bos_id);

DEFINE_string(mode, "requests", "Correctness workload: requests");
DEFINE_int32(requests, 8, "Number of concurrent requests");
DEFINE_int32(prompt_tokens, 512, "Prompt positions including BOS per request");
DEFINE_int32(max_new_tokens, 128, "Exact completion budget per request");
DEFINE_int32(stagger_ms, 0, "Delay between request submissions");
DEFINE_int32(max_decode, 32, "Maximum decodes admitted per step");
DEFINE_int32(timeout_s, 600, "Request workload deadline");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    namespace llm = ::executorch::extension::llm;
    namespace ci = muse_glimmer_ci;
    if (FLAGS_mode != "requests" || FLAGS_pte.empty() ||
        FLAGS_tokenizer.empty() || FLAGS_max_sessions <= 0 ||
        FLAGS_max_session_tokens < 0) {
      throw std::invalid_argument(
          "invalid model or correctness workload flags");
    }
    const ci::RequestOptions options{
        FLAGS_requests,
        FLAGS_prompt_tokens,
        FLAGS_max_new_tokens,
        FLAGS_stagger_ms,
        FLAGS_max_decode,
        FLAGS_timeout_s};
    // Reject impossible workloads before loading tokenizer or model weights.
    ci::validate_requests(
        options, FLAGS_max_sessions, FLAGS_max_session_tokens);
    tokenizers::HFTokenizer tokenizer;
    if (tokenizer.load(FLAGS_tokenizer) != tokenizers::Error::Ok) {
      throw std::runtime_error("could not load the tokenizer");
    }
    llm::MuseGlimmerBackendConfig config;
    config.backend = FLAGS_backend;
    config.model_path = FLAGS_pte;
    config.data_path = FLAGS_data_path;
    config.max_sessions = FLAGS_max_sessions;
    config.max_session_tokens = FLAGS_max_session_tokens;
    config.bos_id = FLAGS_bos_id;
    auto backend = llm::create_muse_glimmer_backend(config);
    if (!backend.ok()) {
      throw std::runtime_error(
          "could not create the executor: error " +
          std::to_string(static_cast<int>(backend.error())));
    }
    ci::EvidenceExecutor executor(*backend->executor);
    llm::serving::ServingRuntimeConfig serving;
    serving.max_sessions = options.requests;
    serving.max_requests = options.requests;
    serving.max_pending_operations = options.requests;
    serving.max_context_length = backend->preparation->max_context_length;
    auto result = ci::run_requests(
        options,
        executor,
        tokenizer,
        serving,
        [spec = backend->preparation](
            const llm::serving::PromptPreparationContext& context,
            llm::serving::PromptInput input) {
          return llm::prepare_muse_glimmer_input(
              context, std::move(input), spec);
        });
    const auto evidence = executor.evidence();
    ci::check_batching(evidence);
    std::cout << "Batching correctness passed: " << result.accepted
              << " requests completed with exact token budgets; "
              << evidence.multi_session_steps
              << " successful multi-session executor steps\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "batching smoke failed: " << error.what() << '\n';
    return 1;
  }
}
