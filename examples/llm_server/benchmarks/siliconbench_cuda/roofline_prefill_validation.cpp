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
    roofline::require(argc >= 4 && argc <= 8, "Usage: roofline_prefill_validation model.pte weights.ptd output.json [history_chunk] [teacher_forced] [dump_logits] [history_tokens]");
    const int history_chunk = argc >= 5 ? std::stoi(argv[4]) : 512;
    const bool teacher_forced = argc >= 6 && std::stoi(argv[5]) != 0;
    const bool dump_logits = argc >= 7 && std::stoi(argv[6]) != 0;
    roofline::require(history_chunk == 512 || history_chunk == 1024, "History chunk must be 512 or 1024");
    const int history_tokens = argc >= 8 ? std::stoi(argv[7]) : 1024;
    roofline::require(history_tokens == 0 || history_tokens == 1024, "History length must be zero or1024");
    struct Dump { std::string file; int rows; int vocab; std::string format; };
    std::vector<Dump> dumps;
    executorch::runtime::runtime_init();
    batching::CudaExecutorConfig cfg;
    cfg.model_path = argv[1]; cfg.data_files = {argv[2]};
    cfg.max_sessions = 32; cfg.max_session_tokens = 2048; cfg.initial_capacity = 512;
    cfg.options.cuda_graph_for_static_methods = true;
    cfg.options.weight_sharing_across_methods = true;
    auto created = batching::CudaExecutor::create(cfg);
    roofline::require(created.ok(), "Executor creation failed");
    auto executor = std::move(created.get());
    roofline::require(executor->initialize(), "Executor initialization failed");
    auto& cache = executor->cache_control();
    std::vector<std::int32_t> owners;
    auto run = [&](const std::vector<roofline::Row>& input, int outputs) {
      batching::RawTargetStep step;
      for (std::size_t i = 0; i < input.size(); ++i) {
        const auto& row = input[i];
        step.tokens.push_back(row.token); step.positions.push_back(row.position);
        step.seq_ids.push_back(owners.at(row.owner));
        if (row.output) step.logit_indices.push_back(i);
      }
      std::vector<std::int64_t> sampled; int callbacks = 0;
      bool ok = executor->execute_raw(step, [&](const batching::RawTargetOutput& output) {
        ++callbacks;
        if (output.selected_indices.empty()) return true;
        auto ids = executor->argmax_rows(output.logits, output.selected_indices.size());
        if (!ids) return false;
        sampled.assign(ids->begin(), ids->end());
        if (dump_logits) {
          const auto& t = output.logits;
          const int rows = static_cast<int>(output.selected_indices.size());
          roofline::require(t.dim() == 2 && t.strides()[1] == 1 && t.strides()[0] == t.size(1), "Logits must be row contiguous");
          std::string format;
          if (t.scalar_type() == executorch::aten::ScalarType::BFloat16) format = "bf16";
          else if (t.scalar_type() == executorch::aten::ScalarType::Half) format = "f16";
          else if (t.scalar_type() == executorch::aten::ScalarType::Float) format = "f32";
          else throw std::runtime_error("Unsupported logits dtype");
          const std::size_t bytes = static_cast<std::size_t>(rows) * t.size(1) * t.element_size();
          std::vector<char> host(bytes);
          roofline::require(cudaMemcpy(host.data(), t.const_data_ptr(), bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "Logit copy failed");
          const std::string file = std::string(argv[3]) + ".logits-" + std::to_string(dumps.size()) + "." + format;
          std::ofstream binary(file, std::ios::binary);
          binary.write(host.data(), host.size()); roofline::require(binary.good(), "Cannot write logits");
          dumps.push_back({file, rows, static_cast<int>(t.size(1)), format});
        }
        return true;
      });
      roofline::require(ok && callbacks == 1, "Expected one successful unsplit raw forward");
      roofline::require(static_cast<int>(sampled.size()) == outputs, "Selected output count differs");
      for (auto id : sampled) roofline::require(id >= 0 && id < executor->vocab_size(), "Invalid sampled token");
      roofline::sync(); return sampled;
    };
    std::ofstream out(argv[3]); roofline::require(out.good(), "Cannot write validation outputs");
    out << "{\"history_chunk\":" << history_chunk << ",\"teacher_forced\":" << (teacher_forced ? "true" : "false") << ",\"history_tokens\":" << history_tokens << ",\"prefix_reuse\":false,\"initial_KV\":512,"
        << "\"runtime_capacity\":2048,\"static_cuda_graph_requested\":true,"
        << "\"timing_used_for_performance\":false,\"cases\":[";
    for (std::size_t i = 0; i < roofline::kMixes.size(); ++i) {
      const auto& mix = roofline::kMixes[i];
      for (auto owner : owners) roofline::require(cache.seq_rm(owner), "Previous owner release failed");
      owners.clear(); int count = mix.decode + (mix.prefill ? 1 : 0);
      for (int owner = 0; owner < count; ++owner) {
        auto id = cache.seq_new(); roofline::require(id.has_value(), "Owner allocation failed"); owners.push_back(*id);
        for (int start = 0; start < history_tokens; start += history_chunk) {
          std::vector<roofline::Row> input;
          for (int pos = start; pos < start + history_chunk; ++pos) input.push_back({roofline::token(owner, pos), pos, owner, false});
          run(input, 0);
        }
      }
      const std::size_t dump_start = dumps.size();
      auto before = executor->method_calls(); auto input = roofline::rows(mix);
      for (auto& row : input) row.position += history_tokens - 1024;
      // The performance harness selects decode outputs only. This separate
      // check also requests the prefill owner's last logits and follows its KV.
      if (mix.prefill) input.back().output = true;
      auto ids = run(input, count); auto after = executor->method_calls();
      std::uint64_t calls = 0;
      if (i) out << ',';
      out << "{\"mix\":\"" << mix.name << "\",\"requested_rows\":" << count << ",\"first_forward_deltas\":{";
      for (std::size_t m = 0; m < after.size(); ++m) {
        if (m) out << ',';
        auto delta = after[m].calls - before[m].calls; calls += delta;
        out << '"' << after[m].name << "\":" << delta;
      }
      roofline::require(calls == 1, "Mixed input was split into more than one forward");
      out << "},\"greedy_steps\":["; roofline::print_tokens(out, ids);
      for (int step = 0; step < 4; ++step) {
        std::vector<roofline::Row> next;
        for (int owner = 0; owner < count; ++owner) {
          int expected = history_tokens + (owner < mix.decode ? 1 : mix.prefill) + step;
          roofline::require(cache.pos(owners[owner]) == expected, "History advancement differs");
          next.push_back({teacher_forced ? roofline::token(owner, expected) : ids[owner], expected, owner, true});
        }
        ids = run(next, count); out << ','; roofline::print_tokens(out, ids);
      }
      out << "],\"logit_dumps\":[";
      for (std::size_t d = dump_start; d < dumps.size(); ++d) {
        if (d > dump_start) out << ',';
        out << "{\"file\":\"" << dumps[d].file << "\",\"rows\":" << dumps[d].rows
            << ",\"vocab\":" << dumps[d].vocab << ",\"format\":\"" << dumps[d].format << "\"}";
      }
      out << "]}"; out.flush();
    }
    out << "],\"completed\":true}\n";
    for (auto owner : owners) roofline::require(cache.seq_rm(owner), "Final owner release failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED " << error.what() << '\n'; return 1;
  }
}
