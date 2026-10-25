/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "roofline_active_common.h"

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/runtime/platform/runtime.h>

namespace batching = executorch::backends::cuda::batching;

int main(int argc, char** argv) {
  try {
    roofline::require(argc == 5, "Usage: roofline_et model.pte weights.ptd output.json repetitions");
    int repetitions = std::stoi(argv[4]);
    roofline::require(repetitions >= 4, "At least four repetitions required");
    executorch::runtime::runtime_init();
    batching::CudaExecutorConfig cfg;
    cfg.model_path = argv[1];
    cfg.data_files = {argv[2]};
    cfg.max_sessions = roofline::kOwners;
    cfg.max_session_tokens = roofline::kCapacity;
    cfg.initial_capacity = 512;
    cfg.options.cuda_graph_for_static_methods = true;
    cfg.options.weight_sharing_across_methods = true;
    auto created = batching::CudaExecutor::create(cfg);
    roofline::require(created.ok(), "CudaExecutor create failed");
    auto executor = std::move(created.get());
    roofline::require(executor->initialize(), "CudaExecutor initialize failed");
    roofline::require(executor->vocab_size() > 190256, "Unexpected MG vocabulary");
    std::vector<std::int32_t> owners;
    auto& cache = executor->cache_control();
    auto run = [&](const std::vector<roofline::Row>& input, int outputs) {
      batching::RawTargetStep step;
      for (std::size_t i = 0; i < input.size(); ++i) {
        const auto& row = input[i];
        step.tokens.push_back(row.token);
        step.positions.push_back(row.position);
        step.seq_ids.push_back(owners.at(row.owner));
        if (row.output) step.logit_indices.push_back(i);
      }
      std::vector<std::int64_t> sampled;
      int callbacks = 0;
      bool ok = executor->execute_raw(step, [&](const batching::RawTargetOutput& output) {
        ++callbacks;
        if (output.selected_indices.empty()) return true;
        auto ids = executor->argmax_rows(output.logits, output.selected_indices.size());
        if (!ids) return false;
        for (auto id : *ids) {
          if (id < 0 || id >= executor->vocab_size()) return false;
          sampled.push_back(id);
        }
        return true;
      });
      roofline::require(ok && callbacks == 1,
                        "Expected exactly one successful raw forward for <=1024 tokens");
      roofline::require(static_cast<int>(sampled.size()) == outputs,
                        "ET selected output count mismatch");
      return sampled;
    };
    auto previous = executor->method_calls();
    auto setup = [&](const roofline::Mix& mix) {
      for (auto owner : owners) roofline::require(cache.seq_rm(owner), "Release prior mix owner failed");
      owners.clear();
      int count = mix.decode + (mix.prefill ? 1 : 0);
      for (int i = 0; i < count; ++i) {
        auto owner = cache.seq_new();
        roofline::require(owner.has_value(), "Requested private owner unavailable");
        owners.push_back(*owner);
      }
      for (int owner = 0; owner < count; ++owner) {
        for (int start = 0; start < roofline::kHistory; start += 512) {
          std::vector<roofline::Row> input;
          for (int pos = start; pos < start + 512; ++pos)
            input.push_back({roofline::token(owner, pos), pos, owner, false});
          run(input, 0);
        }
      }
      roofline::sync();
      previous = executor->method_calls();
    };
    auto rewind = [&] {
      for (auto owner : owners) {
        roofline::require(cache.rewind(owner, roofline::kHistory), "ET rewind failed");
        roofline::require(cache.pos(owner) == roofline::kHistory, "ET context changed");
      }
    };
    std::ofstream out(argv[3]);
    roofline::require(out.good(), "Output file cannot be opened");
    out << "{\"framework\":\"ET\",\"runtime_api\":\"CudaExecutor.execute_raw\","
        << "\"timing\":\"synchronized API forward plus device greedy selected-token return\","
        << "\"history_tokens\":1024,\"maximum_independent_owners\":32,\"history_population\":\"only-current-mix; reset-between-cases\","
        << "\"runtime_context_capacity\":2048,\"model_native_context\":131072,"
        << "\"KV_dtype\":\"BF16\",\"prefix_reuse\":false,"
        << "\"initial_KV_capacity\":512,\"static_cuda_graph_requested\":true,"
        << "\"synthetic_input_definition\":\"roofline_active_common.h::token\","
        << "\"samples_on_device\":" << (executor->samples_on_device() ? "true" : "false") << ',';
    auto evidence = [&](std::ostream& stream, const roofline::Mix& mix, int calls) {
      for (int owner = 0; owner < mix.decode; ++owner) {
        roofline::require(cache.pos(owners[owner]) == roofline::kHistory + 1,
                          "ET decoder history did not advance exactly one token");
      }
      if (mix.prefill) {
        roofline::require(cache.pos(owners[mix.decode]) == roofline::kHistory + mix.prefill,
                          "ET prefill history did not advance by the chunk length");
      }
      auto current = executor->method_calls();
      roofline::require(current.size() == previous.size(), "ET methods changed");
      std::uint64_t sum = 0;
      stream << "\"actual_method_call_deltas\":{";
      for (std::size_t i = 0; i < current.size(); ++i) {
        if (i) stream << ',';
        auto delta = current[i].calls - previous[i].calls;
        sum += delta;
        stream << '"' << current[i].name << "\":" << delta;
      }
      stream << "},\"requested_logit_rows\":" << mix.decode;
      roofline::require(sum == static_cast<std::uint64_t>(calls), "Unexpected ET forward split");
      // Static method logits retain the exported method width. Dynamic methods
      // pad the selector to at least five; no omitted work is hidden in theory.
      int padded = mix.decode;
      int tokens = mix.decode + mix.prefill;
      if (tokens <= 64) {
        padded = 1;
        while (padded < tokens) padded *= 2;
      } else if (padded < 5) {
        padded = 5;
      }
      stream << ",\"exported_or_padded_logit_rows\":" << padded;
      previous = current;
    };
    roofline::measure(out, repetitions, run, rewind, evidence, setup);
    out << ",\"completed\":true}\n";
    out.close();
    for (auto owner : owners) roofline::require(cache.seq_rm(owner), "ET owner release failed");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED " << e.what() << '\n';
    return 1;
  }
}
