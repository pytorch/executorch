/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "../muse_glimmer_batching_workload.h"

#include <iostream>
#include <limits>

namespace ci = muse_glimmer_ci;
namespace batching = ci::batching;
namespace serving = ci::serving;

namespace {
void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("CI workload regression failed");
  }
}

template <typename F>
void rejected(F action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed);
}

class FakeExecutor final : public batching::Executor {
 public:
  bool succeeds = true;
  std::optional<batching::SessionId> open_session() override {
    return 1;
  }
  void close_session(batching::SessionId) override {}
  void set_sampling(
      batching::SessionId,
      const batching::SamplingParams&,
      std::optional<std::uint64_t>) override {}
  bool execute(const batching::BatchInput&, batching::BatchOutput&) override {
    return succeeds;
  }
};

batching::Input input(batching::SessionId session, std::size_t size = 1) {
  return {
      session,
      true,
      0,
      size,
      std::make_shared<const std::vector<batching::Token>>(size, 1000),
      0};
}

void workload_limits() {
  ci::RequestOptions options{5, 600, 64, 200, 32, 600};
  ci::validate_requests(options, 8, 2048);
  rejected([&] { ci::validate_requests(options, 4, 2048); });
  rejected([&] { ci::validate_requests(options, 8, 663); });
  ci::validate_requests(options, 5, 664);
  ci::validate_requests(options, 5);
  rejected([] { ci::check_session_capacity(0, 5); });
  rejected([] { ci::check_session_capacity(1, 0); });
  for (auto member :
       {&ci::RequestOptions::requests,
        &ci::RequestOptions::prompt_tokens,
        &ci::RequestOptions::max_new_tokens,
        &ci::RequestOptions::max_decode,
        &ci::RequestOptions::timeout_s}) {
    auto invalid = options;
    invalid.*member = 0;
    rejected([&] { ci::validate_requests(invalid, 8); });
  }
  options.stagger_ms = -1;
  rejected([&] { ci::validate_requests(options, 8); });
  options = {1, std::numeric_limits<int32_t>::max(), 1, 0, 1, 1};
  rejected([&] {
    ci::validate_requests(options, 1, std::numeric_limits<int32_t>::max());
  });
  const auto prompt = ci::prompt_ids(599, 3);
  require(prompt.size() == 599 && prompt == ci::prompt_ids(599, 3));
  require(prompt != ci::prompt_ids(599, 4));
  require(ci::prompt_ids(0, 0).empty());
  for (const auto token : prompt) {
    require(token >= 1000 && token < 100000);
  }
}

void successful_batch_owners() {
  FakeExecutor inner;
  ci::EvidenceExecutor executor(inner);
  batching::BatchOutput out;
  require(executor.execute({{input(1), input(1)}}, out));
  require(executor.execute({{input(2)}}, out));
  require(executor.evidence().max_sessions == 1);
  rejected([&] { ci::check_batching(executor.evidence()); });
  inner.succeeds = false;
  require(!executor.execute({{input(1), input(2)}}, out));
  require(executor.evidence().multi_session_steps == 0);
  rejected([&] { ci::check_batching(executor.evidence()); });
  // cppcheck-suppress redundantAssignment
  inner.succeeds = true;
  require(executor.execute({{input(1), input(2)}}, out));
  ci::check_batching(executor.evidence());
  require(executor.evidence().multi_session_steps == 1);
  require(executor.evidence().mixed_steps == 0);
  require(executor.execute({{input(1), input(2, 4)}}, out));
  require(executor.evidence().mixed_steps == 1);
}

void exact_completion() {
  const ci::RequestOptions options{2, 600, 64, 0, 2, 1};
  serving::TerminalEvent terminal;
  terminal.finish_reason = serving::FinishReason::Length;
  terminal.stats.prompt_tokens = 600;
  terminal.stats.completion_tokens = 64;
  ci::RequestResult result{2, {terminal, terminal}};
  ci::check_requests_complete(options, result);
  result.accepted = 1;
  rejected([&] { ci::check_requests_complete(options, result); });
  result.accepted = 2;
  result.terminals[1].reset();
  rejected([&] { ci::check_requests_complete(options, result); });
  for (const auto reason :
       {serving::FinishReason::Failed,
        serving::FinishReason::Stop,
        serving::FinishReason::Cancelled}) {
    result.terminals[1] = terminal;
    result.terminals[1]->finish_reason = reason;
    rejected([&] { ci::check_requests_complete(options, result); });
  }
  for (const auto count : {63, 65}) {
    result.terminals[1] = terminal;
    result.terminals[1]->stats.completion_tokens = count;
    rejected([&] { ci::check_requests_complete(options, result); });
  }
  result.terminals[1] = terminal;
  result.terminals[1]->stats.prompt_tokens = 599;
  rejected([&] { ci::check_requests_complete(options, result); });
}
} // namespace

int main() {
  try {
    workload_limits();
    successful_batch_owners();
    exact_completion();
    std::cout << "CI batching workload tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
