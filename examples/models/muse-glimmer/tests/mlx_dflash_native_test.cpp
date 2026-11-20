/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/examples/models/muse-glimmer/runtime/mlx_executor.h>
#include <executorch/extension/llm/batching/executor_utils.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm = executorch::extension::llm;
namespace b = llm::batching;
using Tokens = std::vector<b::Token>;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

b::Input input(b::SessionId sid, int position, Tokens tokens) {
  const auto size = tokens.size();
  return {
      sid,
      true,
      0,
      size,
      std::make_shared<const Tokens>(std::move(tokens)),
      position};
}

std::unique_ptr<b::Executor>
create(const std::string& path, int proposals, bool speculative = true) {
  llm::MuseGlimmerBackendConfig config;
  config.model_path = path;
  config.max_sessions = 4;
  config.use_dflash = speculative;
  config.dflash_n_draft = speculative ? proposals : 0;
  auto backend = llm::MuseGlimmerMLXExecutor::create(config);
  require(backend.ok(), "production factory rejected fixture");
  auto executor = std::move(backend->executor);
  require(executor->initialize(), "executor initialization failed");
  return executor;
}

Tokens
forward(b::Executor& executor, b::SessionId sid, int position, Tokens tokens) {
  executor.set_sampling(sid, {}, 42);
  b::BatchInput batch;
  batch.inputs.push_back(input(sid, position, std::move(tokens)));
  b::BatchOutput output;
  require(executor.execute(batch, output), "native execution failed");
  require(
      output.outputs.size() == 1 && output.outputs[0].has_value(),
      "missing output");
  auto result = output.outputs[0]->tokens;
  require(!result.empty(), "empty completion");
  for (auto token : result) {
    require(token == 7, "verification changed the target's greedy prediction");
  }
  return result;
}

void check_cycles(const std::string& path, int proposals, int rejected) {
  auto executor = create(path, proposals);
  const auto sid = executor->open_session();
  require(sid.has_value(), "session allocation failed");
  const int accepted = rejected < 0 ? proposals : std::min(rejected, proposals);
  const size_t expected = static_cast<size_t>(accepted + 2);
  int position = 0;
  Tokens tokens{1, 2};
  int speculative_cycles = 0, fallback_cycles = 0;
  while (position + static_cast<int>(tokens.size()) <= 32) {
    const int end = position + static_cast<int>(tokens.size());
    auto output = forward(*executor, *sid, position, tokens);
    if (end <= 28) {
      require(
          output.size() == expected, "wrong accepted prefix/correction count");
      ++speculative_cycles;
    } else {
      require(output.size() == 1, "context-tail fallback did not run");
      ++fallback_cycles;
    }
    // Refeeding the last prediction at this position also checks that the
    // executor left correction/bonus uncommitted and discarded rejected K/V.
    position = end + static_cast<int>(output.size()) - 1;
    tokens = {output.back()};
  }
  require(
      speculative_cycles > 1 && fallback_cycles > 0,
      "missing lifecycle coverage");
  executor->close_session(*sid);
  const auto fresh = executor->open_session();
  require(fresh.has_value(), "session reuse failed");
  require(
      forward(*executor, *fresh, 0, {1, 2}).size() == expected,
      "reset changed output");
}

void check_packed_replay(const std::string& path) {
  auto executor = create(path, 3);
  const auto a = executor->open_session(), b = executor->open_session();
  require(a.has_value() && b.has_value(), "packed session allocation failed");
  executor->set_sampling(*a, {}, 42);
  executor->set_sampling(*b, {}, 42);
  b::BatchInput batch;
  batch.inputs = {input(*a, 0, {1, 2}), input(*b, 0, {3, 4, 5})};
  b::BatchOutput output;
  require(executor->execute(batch, output), "packed execution failed");
  require(output.outputs.size() == 2, "missing packed owner");
  for (const auto& result : output.outputs) {
    require(
        result.has_value() && result->tokens == Tokens(5, 7),
        "packed owner mismatch");
  }
  // Discard a speculative suffix as a token-budget stop would, then resume.
  require(forward(*executor, *a, 4, {7}).size() == 5, "rewind/replay failed");
  const auto fork = executor->clone(*a, 8);
  require(fork.has_value(), "prefix clone failed");
  const auto source_output = forward(*executor, *a, 8, {7});
  require(
      forward(*executor, *fork, 8, {7}) == source_output,
      "clone continuation differs");
  executor->close_session(*a);
  require(
      forward(*executor, *b, 7, {7}).size() == 5,
      "closing another owner changed state");
}

void check_temperature(const std::string& path, bool speculative) {
  auto executor = create(path, 3, speculative);
  for (int variant = 0; variant < 4; ++variant) {
    const int trials = variant == 3 ? 128 : 1024;
    b::SamplingParams sampling;
    // The lower-temperature control detects ignoring the requested value.
    sampling.temperature = variant == 3 ? 0.1f : 0.8f;
    sampling.top_k = variant == 1 ? 2 : 0;
    sampling.top_p = variant == 2 ? 0.7f : 1.0f;
    // Controlled head outputs are 32, 31.5, 31 before soft-capping at 64.
    std::array<double, 3> expected{};
    for (size_t token = 0; token < 3; ++token) {
      const double logit = 64 * std::tanh((32 - 0.5 * token) / 64);
      expected[token] =
          std::exp((logit - 64 * std::tanh(0.5)) / sampling.temperature);
    }
    if (variant == 1 || variant == 2) {
      expected[2] = 0;
    }
    const double total = std::accumulate(expected.begin(), expected.end(), 0.0);
    for (auto& probability : expected) {
      probability /= total;
    }
    std::array<std::array<int, 3>, 5> counts{};
    std::array<int, 5> draws{};
    int rejections = 0;
    for (int trial = 0; trial < trials; trial += 4) {
      std::vector<b::SessionId> sessions;
      b::BatchInput batch;
      for (int owner = 0; owner < 4; ++owner) {
        const auto sid = executor->open_session();
        require(sid.has_value(), "temperature session allocation failed");
        executor->set_sampling(*sid, sampling, trial + owner);
        sessions.push_back(*sid);
        batch.inputs.push_back(input(*sid, 0, {1, 2}));
      }
      b::BatchOutput output;
      require(executor->execute(batch, output), "temperature execution failed");
      require(output.outputs.size() == 4, "temperature batch lost an owner");
      for (size_t owner = 0; owner < 4; ++owner) {
        require(
            output.outputs[owner].has_value(), "missing temperature output");
        const auto& tokens = output.outputs[owner]->tokens;
        require(
            speculative ? tokens.size() >= 2 && tokens.size() <= 5
                        : tokens.size() == 1,
            "unexpected temperature output length");
        for (size_t position = 0; position < tokens.size(); ++position) {
          const auto token = tokens[position];
          require(
              token >= 7 && token <= 9,
              "temperature sample outside fixture support");
          require(
              (variant != 1 && variant != 2) || token != 9,
              "target filtering retained an excluded token");
          ++counts[position][token - 7];
          ++draws[position];
        }
        if (speculative) {
          for (size_t position = 1; position + 1 < tokens.size(); ++position) {
            require(
                tokens[position] == 7,
                "accepted draft proposal was not greedy");
          }
          if (tokens.size() < 5) {
            require(
                tokens.back() != 7, "correction sampled the rejected proposal");
          }
          rejections += tokens.size() == 2;
        }
        executor->close_session(sessions[owner]);
      }
    }
    if (speculative && variant != 3) {
      require(
          draws[4] > 50 && rejections > 50,
          "temperature test missed acceptance or rejection");
    }
    // Constant target logits make every reached position independently have p,
    // including anchors, later rejection corrections, and all-accepted bonuses.
    for (int position = 0; position < (speculative ? 5 : 1); ++position) {
      require(
          draws[position] > 0, "temperature test missed an output position");
      for (size_t token = 0; token < 3; ++token) {
        const double variance = expected[token] * (1 - expected[token]);
        require(
            std::abs(
                static_cast<double>(counts[position][token]) / draws[position] -
                expected[token]) <
                5 * std::sqrt(variance / draws[position]) + 0.005,
            "temperature output distribution differs from target");
      }
    }
    std::cout << "temperature=" << sampling.temperature
              << " speculative=" << speculative << " variant=" << variant
              << " anchor_counts=" << counts[0][0] << ',' << counts[0][1] << ','
              << counts[0][2] << " bonuses=" << draws[4]
              << " first_rejections=" << rejections << '\n';
  }
}

std::vector<Tokens>
generate(b::Executor& executor, const std::vector<Tokens>& prompts, int count) {
  std::vector<b::SessionId> sessions;
  std::vector<int> positions(prompts.size(), 0);
  std::vector<Tokens> pending = prompts, completions(prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    auto sid = executor.open_session();
    require(sid.has_value(), "random fixture session allocation failed");
    executor.set_sampling(*sid, {}, 42);
    sessions.push_back(*sid);
  }
  while (true) {
    b::BatchInput batch;
    std::vector<size_t> active;
    for (size_t i = 0; i < prompts.size(); ++i) {
      if (completions[i].size() < static_cast<size_t>(count)) {
        batch.inputs.push_back(input(sessions[i], positions[i], pending[i]));
        active.push_back(i);
      }
    }
    if (active.empty()) {
      break;
    }
    b::BatchOutput output;
    require(executor.execute(batch, output), "random native execution failed");
    require(
        output.outputs.size() == active.size(), "random fixture lost an owner");
    for (size_t row = 0; row < active.size(); ++row) {
      require(output.outputs[row].has_value(), "missing random completion");
      const auto& tokens = output.outputs[row]->tokens;
      require(!tokens.empty(), "empty random completion");
      const size_t i = active[row];
      positions[i] += static_cast<int>(pending[i].size() + tokens.size() - 1);
      pending[i] = {tokens.back()};
      completions[i].insert(completions[i].end(), tokens.begin(), tokens.end());
    }
  }
  for (size_t i = 0; i < prompts.size(); ++i) {
    completions[i].resize(count);
    executor.close_session(sessions[i]);
  }
  return completions;
}

void check_random_continuations(
    const std::string& path,
    const std::vector<int>& proposal_counts) {
  const std::vector<Tokens> prompts{{1, 2, 3}, {9, 5, 11, 2}, {17, 42}};
  std::vector<Tokens> expected;
  {
    auto target = create(path, 0, false);
    for (const auto& prompt : prompts) {
      expected.push_back(generate(*target, {prompt}, 20)[0]);
    }
  }
  if (expected[0] == expected[1]) {
    std::cerr << path << " prompt completions share first token "
              << expected[0][0] << '\n';
  }
  require(
      expected[0] != expected[1],
      "random oracle is insensitive to prompt contents");
  {
    auto target = create(path, 0, false);
    require(
        generate(*target, prompts, 20) == expected,
        "packed target-only continuation differs");
  }
  for (int proposals : proposal_counts) {
    auto executor = create(path, proposals);
    const auto actual = generate(*executor, prompts, 20);
    if (actual != expected) {
      for (size_t owner = 0; owner < actual.size(); ++owner) {
        for (size_t step = 0; step < actual[owner].size(); ++step) {
          if (actual[owner][step] != expected[owner][step]) {
            std::cerr << "proposals=" << proposals << " owner=" << owner
                      << " step=" << step << " actual=" << actual[owner][step]
                      << " expected=" << expected[owner][step] << '\n';
            break;
          }
        }
      }
    }
    require(
        actual == expected,
        "speculative/packed continuation differs from target-only");
    require(
        generate(*executor, prompts, 20) == expected,
        "reset/replay changed random continuation");

    const auto source = executor->open_session();
    require(source.has_value(), "replay session allocation failed");
    executor->set_sampling(*source, {}, 42);
    b::BatchInput initial;
    initial.inputs = {input(*source, 0, prompts[0])};
    b::BatchOutput initial_output;
    require(
        executor->execute(initial, initial_output), "replay prefill failed");
    const int prefix = static_cast<int>(prompts[0].size()) - 1;
    const auto fork = executor->clone(*source, prefix);
    require(fork.has_value(), "random prefix clone failed");
    executor->set_sampling(*fork, {}, 42);
    const std::vector<b::SessionId> replay_sessions{*source, *fork};
    std::vector<int> positions(2, prefix);
    std::vector<Tokens> pending(2, Tokens{prompts[0].back()}), completions(2);
    while (completions[0].size() < 12 || completions[1].size() < 12) {
      b::BatchInput batch;
      std::vector<size_t> active;
      for (size_t i = 0; i < 2; ++i) {
        if (completions[i].size() < 12) {
          batch.inputs.push_back(
              input(replay_sessions[i], positions[i], pending[i]));
          active.push_back(i);
        }
      }
      b::BatchOutput output;
      require(executor->execute(batch, output), "random prefix replay failed");
      require(
          output.outputs.size() == active.size(),
          "prefix replay lost an owner");
      for (size_t row = 0; row < active.size(); ++row) {
        require(
            output.outputs[row].has_value(), "missing prefix replay output");
        const auto& tokens = output.outputs[row]->tokens;
        require(!tokens.empty(), "empty prefix replay output");
        const auto i = active[row];
        positions[i] += static_cast<int>(pending[i].size() + tokens.size() - 1);
        pending[i] = {tokens.back()};
        completions[i].insert(
            completions[i].end(), tokens.begin(), tokens.end());
      }
    }
    for (size_t i = 0; i < 2; ++i) {
      require(
          std::equal(
              completions[i].begin(),
              completions[i].begin() + 12,
              expected[0].begin()),
          "rewound/cloned continuation differs from target-only replay");
      executor->close_session(replay_sessions[i]);
    }
  }
}

std::vector<std::vector<Tokens>> routing_trace(
    b::Executor& executor,
    const std::vector<Tokens>& prompts,
    bool reverse) {
  std::vector<b::SessionId> sessions;
  for (size_t i = 0; i < prompts.size(); ++i) {
    const auto sid = executor.open_session();
    require(sid.has_value(), "routing session allocation failed");
    executor.set_sampling(*sid, {}, 42);
    sessions.push_back(*sid);
  }
  std::vector<int> positions(prompts.size(), 0);
  auto pending = prompts;
  std::vector<std::vector<Tokens>> traces(prompts.size());
  for (int round = 0; round < 4; ++round) {
    std::vector<size_t> owners(prompts.size());
    std::iota(owners.begin(), owners.end(), 0);
    if (reverse != (round % 2 == 1)) {
      std::reverse(owners.begin(), owners.end());
    }
    b::BatchInput batch;
    for (auto owner : owners) {
      batch.inputs.push_back(
          input(sessions[owner], positions[owner], pending[owner]));
    }
    b::BatchOutput output;
    require(executor.execute(batch, output), "routing execution failed");
    require(output.outputs.size() == owners.size(), "routing lost an owner");
    for (size_t row = 0; row < owners.size(); ++row) {
      const auto owner = owners[row];
      const auto& result = output.outputs[row];
      require(result && !result->tokens.empty(), "missing routing output");
      const auto& tokens = result->tokens;
      require(
          tokens == Tokens(tokens.size(), prompts[owner].back()),
          "routing changed verified target tokens");
      traces[owner].push_back(tokens);
      positions[owner] +=
          static_cast<int>(pending[owner].size() + tokens.size() - 1);
      pending[owner] = {tokens.back()};
    }
  }
  for (auto sid : sessions) {
    executor.close_session(sid);
  }
  return traces;
}

void check_draft_routing(const std::string& path) {
  // Verification repairs bad proposals, so compare complete rounds, not text
  // alone.
  const std::vector<Tokens> prompts{{8, 7}, {7, 7, 7, 8}};
  for (int proposals : {1, 2, 3}) {
    auto executor = create(path, proposals);
    std::vector<std::vector<Tokens>> expected;
    for (const auto& prompt : prompts) {
      auto trace = routing_trace(*executor, {prompt}, false)[0];
      require(
          trace[0].size() >= 3, "routing fixture rejected its first proposal");
      expected.push_back(std::move(trace));
    }
    for (bool reverse : {false, true}) {
      const auto actual = routing_trace(*executor, prompts, reverse);
      for (size_t owner = 0; owner < prompts.size(); ++owner) {
        for (size_t round = 0; round < expected[owner].size(); ++round) {
          if (actual[owner][round] != expected[owner][round]) {
            std::cerr << path << " proposals=" << proposals
                      << " owner=" << owner << " round=" << round
                      << " packed=" << actual[owner][round].size()
                      << " solo=" << expected[owner][round].size() << '\n';
            throw std::runtime_error(
                "packed draft acceptance differs from solo");
          }
        }
      }
    }
  }
}

void check_selector_metadata(const std::string& path) {
  executorch::extension::Module module(path);
  const auto selector = module.execute("get_dflash_selector_top_k");
  require(
      selector.ok() && selector->size() == 1 && selector->at(0).isInt() &&
          selector->at(0).toInt() == 3,
      "nonconstant selector fixture must return a valid top-k");
  llm::MuseGlimmerBackendConfig config;
  config.model_path = path;
  config.use_dflash = true;
  const auto backend = llm::MuseGlimmerMLXExecutor::create(config);
  require(
      !backend.ok() &&
          backend.error() == executorch::runtime::Error::InvalidProgram,
      "nonconstant selector metadata was accepted");
}

void check_owner_proposals(const std::string& path) {
  for (int proposals : {1, 2, 3}) {
    for (bool reverse : {false, true}) {
      auto executor = create(path, proposals);
      std::vector<b::SessionId> sessions;
      for (int i = 0; i < 2; ++i) {
        const auto sid = executor->open_session();
        require(sid.has_value(), "owner proposal session allocation failed");
        executor->set_sampling(*sid, {}, 42);
        sessions.push_back(*sid);
      }
      b::BatchInput batch;
      batch.inputs = {
          input(sessions[0], 0, {1, 7}), input(sessions[1], 0, {2, 3, 8})};
      if (reverse) {
        std::reverse(batch.inputs.begin(), batch.inputs.end());
      }
      b::BatchOutput output;
      require(executor->execute(batch, output), "owner proposals failed");
      require(output.outputs.size() == 2, "owner proposals lost a session");
      for (int row = 0; row < 2; ++row) {
        const int owner = reverse ? 1 - row : row;
        const int accepted = std::min(1 + 2 * owner, proposals);
        require(
            output.outputs[row] &&
                output.outputs[row]->tokens == Tokens(accepted + 2, 7 + owner),
            "owner-specific accepted prefix/correction count differs");
      }
      for (auto sid : sessions) {
        executor->close_session(sid);
      }
    }
  }
}

void check_draft_numerics(const std::string& directory, int proposals) {
  namespace cache = llm::cache;
  namespace rt = executorch::runtime;
  namespace aten = executorch::aten;
  using executorch::extension::make_tensor_ptr;
  std::ifstream stream(directory + "/draft_reference.json");
  const auto reference = nlohmann::json::parse(stream);
  const int block = reference["block_size"], top_k = reference["top_k"],
            hidden_size = reference["hidden_size"];
  require(proposals > 0 && proposals < block, "invalid oracle proposal count");
  cache::CacheGeometry geometry;
  geometry.dflash_block_size = block;
  for (const auto& layer : reference["layers"]) {
    const int window = layer["window"];
    geometry.layers.push_back(
        {{window ? cache::LayerPolicy::Kind::Ring
                 : cache::LayerPolicy::Kind::Flat,
          window},
         layer["kv_heads"],
         layer["head_dim"]});
  }
  cache::CacheConfig config{};
  config.capacity = 4 * reference["max_seq_len"].get<int>();
  config.max_write = reference["max_seq_len"];
  config.kv_dtype = static_cast<int>(aten::ScalarType::Half);
  auto owner = cache::CacheFactory::global().build(
      "MLXBackend", cache::kind::kBatched, geometry, config);
  require(owner.ok(), "draft oracle cache creation failed");
  cache::InstallGuard guard(*owner);
  auto* control = (*owner)->as<cache::BatchControl>();
  std::vector<int32_t> sessions;
  for (int i = 0; i < 2; ++i) {
    const auto sid = control->seq_new();
    require(sid.has_value(), "draft oracle session allocation failed");
    sessions.push_back(*sid);
  }
  executorch::extension::Module module(directory + "/model.pte");
  rt::BackendOptions<1> options;
  require(
      b::load_method_with_cache(
          module, "draft_forward", "MLXBackend", guard, options) ==
          rt::Error::Ok,
      "draft oracle method load failed");
  for (const auto& step : reference["steps"]) {
    if (step["clone"].get<bool>()) {
      const auto cloned = control->seq_clone(sessions[0], std::nullopt);
      require(cloned.has_value(), "draft oracle clone failed");
      sessions.push_back(*cloned);
    }
    const auto& rows = step["rows"];
    const int owners = static_cast<int>(rows.size());
    int context = 0;
    std::vector<int64_t> token_ids, positions, order, expected_ids;
    std::vector<float> expected_scores;
    std::vector<aten::Half> hidden;
    std::vector<int32_t> sequence_ids;
    for (const auto& row : rows) {
      const int count = row["context"], position = row["position"];
      const auto tokens = row["tokens"].get<std::vector<int64_t>>();
      token_ids.insert(token_ids.end(), tokens.begin(), tokens.end());
      const auto values = row["hidden"].get<std::vector<float>>();
      hidden.insert(hidden.end(), values.begin(), values.end());
      for (int p = 0; p < count; ++p)
        positions.push_back(position + p);
      context += count;
      const auto ids = row["ids"].get<std::vector<int64_t>>();
      expected_ids.insert(
          expected_ids.end(), ids.begin(), ids.begin() + proposals * top_k);
      const auto scores = row["scores"].get<std::vector<float>>();
      expected_scores.insert(
          expected_scores.end(),
          scores.begin(),
          scores.begin() + proposals * top_k * top_k);
    }
    int offset = 0;
    for (int owner_index = 0; owner_index < owners; ++owner_index) {
      const auto& row = rows[owner_index];
      const int count = row["context"], position = row["position"],
                owner_index_in_fixture = row["owner"];
      sequence_ids.insert(
          sequence_ids.end(), count + block, sessions[owner_index_in_fixture]);
      for (int p = 0; p < count; ++p)
        order.push_back(offset++);
      for (int p = 0; p < block; ++p) {
        positions.push_back(position + count + p);
        order.push_back(context + owner_index * block + p);
      }
    }
    auto tokens = make_tensor_ptr({owners, block}, std::move(token_ids));
    auto hidden_tensor =
        make_tensor_ptr({1, context, hidden_size}, std::move(hidden));
    auto position_tensor =
        make_tensor_ptr({context + owners * block}, std::move(positions));
    auto order_tensor =
        make_tensor_ptr({context + owners * block}, std::move(order));
    std::vector<int64_t> proposal_indices(proposals, -7);
    auto proposal_tensor =
        make_tensor_ptr({proposals}, std::move(proposal_indices));
    require(
        control->declare_step(sequence_ids), "draft oracle admission failed");
    auto output = module.execute(
        "draft_forward",
        {*tokens,
         *hidden_tensor,
         *position_tensor,
         *order_tensor,
         *proposal_tensor});
    if (!output.ok()) {
      std::cerr << directory << " owners=" << owners << " context=" << context
                << " error=" << static_cast<int>(output.error()) << '\n';
    }
    require(
        output.ok() && output->size() == 2, "draft oracle execution failed");
    const auto ids = output->at(0).toTensor();
    const auto scores = output->at(1).toTensor();
    require(
        ids.numel() == expected_ids.size() &&
            scores.numel() == expected_scores.size(),
        "draft oracle output shape mismatch");
    // FP16 near-ties can reorder top-k; compare the same token pair on both
    // score axes while requiring exactly the same candidate set at each step.
    std::vector<int> permutation(expected_ids.size());
    for (size_t i = 0; i < expected_ids.size(); ++i) {
      const auto begin = expected_ids.begin() + i / top_k * top_k;
      const auto found =
          std::find(begin, begin + top_k, ids.const_data_ptr<int64_t>()[i]);
      require(
          found != begin + top_k,
          "native draft candidates differ from eager reference");
      permutation[i] = static_cast<int>(found - begin);
      for (size_t previous = i / top_k * top_k; previous < i; ++previous) {
        require(
            permutation[i] != permutation[previous],
            "native draft candidates contain duplicates");
      }
    }
    for (int row = 0; row < owners; ++row) {
      for (int proposal = 0; proposal < proposals; ++proposal) {
        for (int previous = 0; previous < top_k; ++previous) {
          for (int current = 0; current < top_k; ++current) {
            const int previous_ref = proposal == 0
                ? previous
                : permutation
                      [(row * proposals + proposal - 1) * top_k + previous];
            const int current_ref =
                permutation[(row * proposals + proposal) * top_k + current];
            const int base = (row * proposals + proposal) * top_k * top_k;
            const auto expected =
                expected_scores[base + previous_ref * top_k + current_ref];
            const auto actual =
                scores
                    .const_data_ptr<float>()[base + previous * top_k + current];
            require(
                std::isfinite(actual) &&
                    std::abs(actual - expected) <=
                        0.002f + 0.01f * std::abs(expected),
                "native draft scores differ from eager reference");
          }
        }
      }
    }
    for (const auto& row : rows) {
      const int index = row["owner"], position = row["position"],
                count = row["context"];
      require(
          control->pos(sessions[index]) == position + count,
          "temporary draft rows persisted");
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  try {
    require(argc == 2, "usage: muse_glimmer_mlx_dflash_test FIXTURE_DIRECTORY");
    const std::string root = argv[1];
    check_selector_metadata(root + "/nonconstant_selector.pte");
    {
      auto executor = create(root + "/single_proposal/model.pte", 1);
      const auto sid = executor->open_session();
      require(sid.has_value(), "single proposal session allocation failed");
      require(
          forward(*executor, *sid, 0, {1, 2}).size() == 3,
          "fixed single-proposal selector lost its input ABI");
      executor->close_session(*sid);
    }
    for (int rejected : {-1, 0, 1, 2}) {
      const auto path =
          root + "/reject_" + std::to_string(rejected) + "/model.pte";
      for (int proposals : {1, 2, 3}) {
        check_cycles(path, proposals, rejected);
      }
    }
    check_packed_replay(root + "/reject_-1/model.pte");
    check_owner_proposals(root + "/owner_sensitive/model.pte");
    check_draft_routing(root + "/routing/model.pte");
    check_draft_routing(root + "/routing_legacy/model.pte");
    for (int proposals = 1; proposals < 4; ++proposals) {
      check_draft_numerics(root + "/random", proposals);
    }
    check_random_continuations(root + "/random/model.pte", {1, 2, 3});
    for (int proposals = 1; proposals < 16; ++proposals) {
      check_draft_numerics(root + "/quantized", proposals);
    }
    check_random_continuations(root + "/quantized/model.pte", {1, 7, 15});
    check_cycles(root + "/legacy/model.pte", 3, -1);
    check_temperature(root + "/reject_-1/model.pte", false);
    check_temperature(root + "/reject_-1/model.pte", true);
    check_temperature(root + "/legacy/model.pte", true);
    auto target = create(root + "/reject_-1/model.pte", 0, false);
    const auto sid = target->open_session();
    require(sid.has_value(), "target session allocation failed");
    require(
        forward(*target, *sid, 0, {1, 2}).size() == 1,
        "target-only path speculated");

    llm::MuseGlimmerBackendConfig invalid;
    invalid.model_path = root + "/reject_-1/model.pte";
    invalid.use_dflash = true;
    invalid.dflash_n_draft = 4;
    require(
        llm::MuseGlimmerMLXExecutor::create(invalid).error() ==
            executorch::runtime::Error::InvalidArgument,
        "out-of-range proposal count was accepted");
    std::cout
        << "Native MLX DFlash verification, fallback, replay, and legacy checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
