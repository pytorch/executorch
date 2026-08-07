/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/gemma4/runner/gemma4_spec_runner.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace executorch::examples::gemma4 {
namespace {

constexpr int64_t kVocabSize = 262144;
const std::vector<int64_t> kStopTokens = {1, 106, 50};
const std::vector<int64_t> kNoStopTokens = {};

// Two accepted drafts plus the bonus: the widest well-formed K=2 round.
Gemma4K2Output FullMatch() {
  return Gemma4K2Output{{10, 11}, {10, 11, 90}, 2, 90, 0.0f};
}

bool Rejected(const Gemma4K2Output& output) {
  return !reconcile_gemma4_k2(output, 2, 3, kStopTokens).valid;
}

TEST(Gemma4SpecControllerTest, AdvancesByAcceptedPrefixAndSeedsFromBonus) {
  const auto no_match = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {90, 91, 92}, 0, 90, 0.0f}, 2, 3, kStopTokens);
  EXPECT_TRUE(no_match.valid);
  EXPECT_EQ(no_match.committed, std::vector<int64_t>({90}));
  EXPECT_EQ(no_match.selected, std::vector<int64_t>({90}));
  EXPECT_EQ(no_match.discarded, std::vector<int64_t>({}));
  EXPECT_EQ(no_match.accepted_drafts, 0u);
  EXPECT_EQ(no_match.next_position, 3);
  EXPECT_EQ(no_match.next_seed, 90);
  EXPECT_FALSE(no_match.stopped);

  const auto one_match = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {10, 90, 92}, 1, 90, 0.0f}, 2, 3, kStopTokens);
  EXPECT_TRUE(one_match.valid);
  EXPECT_EQ(one_match.committed, std::vector<int64_t>({10, 90}));
  EXPECT_EQ(one_match.selected, std::vector<int64_t>({10, 90}));
  EXPECT_EQ(one_match.accepted_drafts, 1u);
  EXPECT_EQ(one_match.next_position, 4);
  EXPECT_EQ(one_match.next_seed, 90);

  const auto two_matches = reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens);
  EXPECT_TRUE(two_matches.valid);
  EXPECT_EQ(two_matches.committed, std::vector<int64_t>({10, 11, 90}));
  EXPECT_EQ(two_matches.selected, std::vector<int64_t>({10, 11, 90}));
  EXPECT_EQ(two_matches.accepted_drafts, 2u);
  EXPECT_EQ(two_matches.next_position, 5);
  EXPECT_EQ(two_matches.next_seed, 90);
}

TEST(Gemma4SpecControllerTest, ChainedRoundsWalkStartPositionsTwoThreeFive) {
  const auto first = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {90, 91, 92}, 0, 90, 0.0f}, 2, 8, kStopTokens);
  ASSERT_TRUE(first.valid);
  EXPECT_EQ(first.next_position, 3);
  EXPECT_EQ(first.next_seed, 90);

  const auto second = reconcile_gemma4_k2(
      Gemma4K2Output{{20, 21}, {20, 91, 92}, 1, 91, 0.0f},
      first.next_position,
      8,
      kStopTokens);
  ASSERT_TRUE(second.valid);
  EXPECT_EQ(second.next_position, 5);
  EXPECT_EQ(second.next_seed, 91);

  const auto third = reconcile_gemma4_k2(
      Gemma4K2Output{{30, 31}, {30, 31, 92}, 2, 92, 0.0f},
      second.next_position,
      8,
      kStopTokens);
  ASSERT_TRUE(third.valid);
  EXPECT_EQ(third.next_position, 8);
  EXPECT_EQ(third.next_seed, 92);
}

TEST(Gemma4SpecControllerTest, SeedsFromBonusNotFromDraftOrTargetTail) {
  const auto decision = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {10, 90, 92}, 1, 90, 0.0f}, 2, 3, kStopTokens);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.next_seed, 90);
  EXPECT_NE(decision.next_seed, 10);
  EXPECT_NE(decision.next_seed, 11);
  EXPECT_NE(decision.next_seed, 92);
  EXPECT_EQ(decision.selected.back(), decision.next_seed);
}

TEST(Gemma4SpecControllerTest, TruncatesAtStopWithoutCommittingTheStopToken) {
  const auto decision = reconcile_gemma4_k2(
      Gemma4K2Output{{106, 11}, {106, 11, 90}, 2, 90, 0.0f}, 2, 3, kStopTokens);

  EXPECT_TRUE(decision.valid);
  EXPECT_TRUE(decision.stopped);
  EXPECT_EQ(decision.stop_token, 106);
  EXPECT_TRUE(decision.committed.empty());
  EXPECT_EQ(decision.discarded, std::vector<int64_t>({11, 90}));
  EXPECT_EQ(decision.next_position, 5);
  EXPECT_EQ(decision.next_seed, 90);
}

TEST(Gemma4SpecControllerTest, StopTokenInBonusSlotCommitsAcceptedPrefix) {
  const auto decision = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {10, 11, 1}, 2, 1, 0.0f}, 2, 3, kStopTokens);

  EXPECT_TRUE(decision.valid);
  EXPECT_TRUE(decision.stopped);
  EXPECT_EQ(decision.stop_token, 1);
  EXPECT_EQ(decision.committed, std::vector<int64_t>({10, 11}));
  EXPECT_TRUE(decision.discarded.empty());
}

TEST(Gemma4SpecControllerTest, TruncatesAtBudgetWithoutAnotherRound) {
  const auto one = reconcile_gemma4_k2(FullMatch(), 2, 1, kStopTokens);
  EXPECT_TRUE(one.valid);
  EXPECT_FALSE(one.stopped);
  EXPECT_EQ(one.committed, std::vector<int64_t>({10}));
  EXPECT_EQ(one.discarded, std::vector<int64_t>({11, 90}));

  const auto two = reconcile_gemma4_k2(FullMatch(), 2, 2, kStopTokens);
  EXPECT_EQ(two.committed, std::vector<int64_t>({10, 11}));
  EXPECT_EQ(two.discarded, std::vector<int64_t>({90}));

  const auto three = reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens);
  EXPECT_EQ(three.committed, std::vector<int64_t>({10, 11, 90}));
  EXPECT_TRUE(three.discarded.empty());
}

TEST(Gemma4SpecControllerTest, RejectsInconsistentGraphMatchMetadata) {
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {10, 91, 92}, 2, 92, 0.0f}));
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {90, 91, 92}, 1, 91, 0.0f}));
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {10, 11, 90}, 1, 11, 0.0f}));
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {10, 90, 92}, 0, 10, 0.0f}));
}

TEST(Gemma4SpecControllerTest, RejectsBonusThatIsNotGreedyAtMatchCount) {
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {10, 11, 90}, 2, 11, 0.0f}));
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {90, 91, 92}, 0, 91, 0.0f}));
  EXPECT_TRUE(Rejected(Gemma4K2Output{{10, 11}, {10, 90, 92}, 1, 92, 0.0f}));
}

TEST(Gemma4SpecControllerTest, RejectsStartPositionBelowTwo) {
  for (const int64_t start : {-1, 0, 1}) {
    EXPECT_FALSE(reconcile_gemma4_k2(FullMatch(), start, 3, kStopTokens).valid)
        << "start_position=" << start;
  }
  EXPECT_TRUE(reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens).valid);
}

TEST(Gemma4SpecControllerTest, RejectsZeroTokenBudget) {
  EXPECT_FALSE(reconcile_gemma4_k2(FullMatch(), 2, 0, kStopTokens).valid);
  EXPECT_TRUE(reconcile_gemma4_k2(FullMatch(), 2, 1, kStopTokens).valid);
}

TEST(Gemma4SpecControllerTest, RejectsNonPositiveVocabSize) {
  EXPECT_FALSE(reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens, 0).valid);
  EXPECT_FALSE(reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens, -1).valid);
  EXPECT_TRUE(reconcile_gemma4_k2(FullMatch(), 2, 3, kStopTokens, 91).valid);
}

TEST(Gemma4SpecControllerTest, RejectsMatchCountOutsideZeroToTwo) {
  Gemma4K2Output low = FullMatch();
  low.match_count = -1;
  EXPECT_FALSE(reconcile_gemma4_k2(low, 2, 3, kStopTokens).valid);

  Gemma4K2Output high = FullMatch();
  high.match_count = 3;
  EXPECT_FALSE(reconcile_gemma4_k2(high, 2, 3, kStopTokens).valid);
}

TEST(Gemma4SpecControllerTest, RejectsNonFiniteStateProbe) {
  for (const float probe :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    Gemma4K2Output output = FullMatch();
    output.state_probe = probe;
    EXPECT_FALSE(reconcile_gemma4_k2(output, 2, 3, kStopTokens).valid);
  }
  Gemma4K2Output finite = FullMatch();
  finite.state_probe = -3.5f;
  EXPECT_TRUE(reconcile_gemma4_k2(finite, 2, 3, kStopTokens).valid);
}

TEST(Gemma4SpecControllerTest, RejectsOutOfRangeTokenIds) {
  Gemma4K2Output low_candidate = FullMatch();
  low_candidate.candidates = {-1, 11};
  EXPECT_FALSE(reconcile_gemma4_k2(low_candidate, 2, 3, kStopTokens).valid);

  Gemma4K2Output high_candidate = FullMatch();
  high_candidate.candidates = {kVocabSize, 11};
  EXPECT_FALSE(reconcile_gemma4_k2(high_candidate, 2, 3, kStopTokens).valid);

  Gemma4K2Output low_greedy = FullMatch();
  low_greedy.target_greedy = {10, 11, -1};
  EXPECT_FALSE(reconcile_gemma4_k2(low_greedy, 2, 3, kStopTokens).valid);

  Gemma4K2Output high_greedy = FullMatch();
  high_greedy.target_greedy = {10, 11, kVocabSize};
  high_greedy.bonus = kVocabSize;
  EXPECT_FALSE(reconcile_gemma4_k2(high_greedy, 2, 3, kStopTokens).valid);

  Gemma4K2Output narrow_vocab = FullMatch();
  EXPECT_FALSE(reconcile_gemma4_k2(narrow_vocab, 2, 3, kStopTokens, 90).valid);
}

TEST(Gemma4SpecControllerTest, EmptyStopTokenListNeverStops) {
  const auto decision = reconcile_gemma4_k2(
      Gemma4K2Output{{106, 1}, {106, 1, 90}, 2, 90, 0.0f}, 2, 3, kNoStopTokens);
  EXPECT_TRUE(decision.valid);
  EXPECT_FALSE(decision.stopped);
  EXPECT_EQ(decision.stop_token, -1);
  EXPECT_EQ(decision.committed, std::vector<int64_t>({106, 1, 90}));
}

TEST(Gemma4SpecControllerTest, RejectedDecisionKeepsDocumentedDefaults) {
  const auto decision = reconcile_gemma4_k2(
      Gemma4K2Output{{10, 11}, {10, 91, 92}, 2, 92, 0.0f}, 2, 3, kStopTokens);
  EXPECT_FALSE(decision.valid);
  EXPECT_FALSE(decision.stopped);
  EXPECT_EQ(decision.stop_token, -1);
  EXPECT_EQ(decision.next_position, -1);
  EXPECT_EQ(decision.next_seed, -1);
  EXPECT_EQ(decision.accepted_drafts, 0u);
  EXPECT_TRUE(decision.selected.empty());
  EXPECT_TRUE(decision.committed.empty());
  EXPECT_TRUE(decision.discarded.empty());
}

TEST(Gemma4SpecControllerTest, DefaultVocabSizeMatchesTheExportContract) {
  Gemma4K2Output output = FullMatch();
  output.candidates = {kVocabSize - 1, 11};
  output.target_greedy = {kVocabSize - 1, 11, 90};
  EXPECT_TRUE(reconcile_gemma4_k2(output, 2, 3, kStopTokens).valid);

  output.candidates = {kVocabSize, 11};
  output.target_greedy = {kVocabSize, 11, 90};
  EXPECT_FALSE(reconcile_gemma4_k2(output, 2, 3, kStopTokens).valid);
}

TEST(Gemma4SpecControllerTest, ConfigDefaultsMatchTheExportContract) {
  const Gemma4SpecRunnerConfig config;
  EXPECT_EQ(config.vocab_size, kVocabSize);
  EXPECT_EQ(config.max_input_length, 512);
  EXPECT_EQ(config.target_capacity, 8960);
  EXPECT_EQ(config.donor_capacity, 8960);
  EXPECT_EQ(config.method_name, "k2_round");
}

using ::executorch::runtime::Error;

TEST(Gemma4SpecRunnerLifecycleTest, FreshRunnerReportsEmptyAccounting) {
  Gemma4SpecRunner runner;
  EXPECT_FALSE(runner.is_loaded());
  EXPECT_EQ(runner.execute_count(), 0u);
  EXPECT_EQ(runner.accepted_drafts(), 0u);
  EXPECT_EQ(runner.buffered_tokens(), 0u);
}

TEST(Gemma4SpecRunnerLifecycleTest, ResetWithoutAModuleIsInvalidState) {
  Gemma4SpecRunner runner;
  EXPECT_EQ(runner.reset(), Error::InvalidState);
  EXPECT_EQ(runner.reset(), Error::InvalidState);
  EXPECT_FALSE(runner.is_loaded());
  EXPECT_EQ(runner.buffered_tokens(), 0u);
  EXPECT_EQ(runner.execute_count(), 0u);
  EXPECT_EQ(runner.accepted_drafts(), 0u);
}

TEST(Gemma4SpecRunnerLifecycleTest, UnloadWithoutAModuleSucceeds) {
  Gemma4SpecRunner runner;
  EXPECT_EQ(runner.unload(), Error::Ok);
  EXPECT_FALSE(runner.is_loaded());
}

TEST(Gemma4SpecRunnerLifecycleTest, LoadRejectsEmptyPathAndInvalidConfig) {
  Gemma4SpecRunner runner;
  EXPECT_EQ(runner.load("", {}), Error::InvalidArgument);

  Gemma4SpecRunnerConfig config;
  config.vocab_size = 0;
  Gemma4SpecRunner zero_vocab(config);
  EXPECT_EQ(zero_vocab.load("model.pte", {}), Error::InvalidArgument);

  Gemma4SpecRunnerConfig unnamed;
  unnamed.method_name = "";
  Gemma4SpecRunner no_method(unnamed);
  EXPECT_EQ(no_method.load("model.pte", {}), Error::InvalidArgument);
}

// Earlier empty-PTD cases cannot isolate a config clause. These pass exactly
// three PTDs and vary one clause at a time.
TEST(Gemma4SpecRunnerLifecycleTest, LoadRejectsEachInvalidConfigClause) {
  const auto rejects = [](const char* clause,
                          const Gemma4SpecRunnerConfig& config) {
    Gemma4SpecRunner runner(config);
    EXPECT_EQ(
        runner.load("k2_round.pte", {"a.ptd", "b.ptd", "c.ptd"}),
        Error::InvalidArgument)
        << clause;
    EXPECT_FALSE(runner.is_loaded()) << clause;
  };
  const int64_t above_int32 =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  for (const int64_t vocab : {int64_t{0}, int64_t{-1}}) {
    Gemma4SpecRunnerConfig config;
    config.vocab_size = vocab;
    rejects("vocab_size", config);
  }
  for (const int64_t length : {int64_t{0}, int64_t{-1}, above_int32}) {
    Gemma4SpecRunnerConfig config;
    config.max_input_length = length;
    rejects("max_input_length", config);
  }
  for (const int64_t capacity : {int64_t{0}, int64_t{-1}}) {
    Gemma4SpecRunnerConfig target;
    target.target_capacity = capacity;
    rejects("target_capacity", target);
    Gemma4SpecRunnerConfig donor;
    donor.donor_capacity = capacity;
    rejects("donor_capacity", donor);
  }
  for (const char* name : {"", "k2"}) {
    Gemma4SpecRunnerConfig config;
    config.method_name = name;
    rejects("method_name", config);
  }
}

TEST(Gemma4SpecRunnerLifecycleTest, LoadRejectsAnyPtdCountOtherThanThree) {
  const std::vector<std::vector<std::string>> wrong = {
      {}, {"a.ptd"}, {"a.ptd", "b.ptd"}, {"a.ptd", "b.ptd", "c.ptd", "d.ptd"}};
  for (const auto& ptds : wrong) {
    Gemma4SpecRunner runner;
    EXPECT_EQ(runner.load("k2_round.pte", ptds), Error::InvalidArgument)
        << ptds.size() << " PTDs";
  }
}

TEST(Gemma4SpecRunnerLifecycleTest, WellFormedLoadIsNotRejectedByArgumentFence) {
  Gemma4SpecRunner runner;
  const Error error =
      runner.load("no_such_k2_round.pte", {"a.ptd", "b.ptd", "c.ptd"});

  EXPECT_NE(error, Error::InvalidArgument);
  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(runner.is_loaded());
}
TEST(Gemma4SpecRunnerLifecycleTest, UnloadedRunnerRejectsEveryStepEntry) {
  Gemma4SpecRunner runner;
  EXPECT_EQ(
      runner.execute({10}, {2}, false, 2).error(), Error::InvalidState);
  EXPECT_EQ(runner.prefill_step(10, 0), Error::InvalidArgument);
  EXPECT_EQ(runner.step(10, 2).error(), Error::InvalidArgument);
}

TEST(Gemma4SpecRunnerLifecycleTest, RejectedExecutionsBillNothingAcrossReset) {
  Gemma4SpecRunner runner;
  const auto first = runner.execute({10, 11}, {2, 3}, false, 2);
  EXPECT_EQ(first.error(), Error::InvalidState);
  EXPECT_EQ(runner.execute_count(), 0u);

  EXPECT_EQ(runner.reset(), Error::InvalidState);

  const auto second = runner.execute({10, 11}, {2, 3}, false, 2);
  EXPECT_EQ(second.error(), first.error());
  EXPECT_EQ(runner.execute_count(), 0u);
  EXPECT_EQ(runner.accepted_drafts(), 0u);
  EXPECT_EQ(runner.buffered_tokens(), 0u);
  EXPECT_FALSE(runner.is_loaded());
}

TEST(Gemma4SpecRunnerLifecycleTest, GenerateRejectsMalformedRequests) {
  Gemma4SpecRunner runner;
  EXPECT_EQ(runner.generate({}, 4, {}).error(), Error::InvalidArgument);
  EXPECT_EQ(runner.generate({10, 11}, 0, {}).error(), Error::InvalidArgument);
  EXPECT_EQ(runner.generate({10}, 2, {}).error(), Error::InvalidArgument);
  EXPECT_EQ(
      runner.generate({10, 11}, 4, {kVocabSize}).error(),
      Error::InvalidArgument);
  EXPECT_EQ(runner.generate({10, 11}, 4, {-1}).error(), Error::InvalidArgument);
  EXPECT_EQ(runner.generate({10, 11}, 4, {}).error(), Error::InvalidState);
}

TEST(Gemma4SpecRunnerLifecycleTest, ProfileJsonIsSchemaVersionOne) {
  Gemma4SpecRunner runner;
  runner.set_profiling_enabled(true);
  const std::string profile = runner.profile_json();
  EXPECT_NE(profile.find("\"schemaVersion\":1"), std::string::npos);
  EXPECT_NE(profile.find("\"execute_generation\":0"), std::string::npos);
  runner.set_profiling_enabled(false);
  EXPECT_EQ(runner.profile_json(), profile);
}

} // namespace
} // namespace executorch::examples::gemma4
