/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "api/NeuronAdapterLibrarySelector.h"

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace executorch {
namespace backends {
namespace neuron {
namespace {

struct FakeLibrary {
  std::string name;
  int major;
};

class AdapterLibrarySelectorTest : public ::testing::Test {
 protected:
  void AddLibrary(const std::string& name, int major) {
    libraries_.emplace(name, FakeLibrary{name, major});
  }

  AdapterLibrarySelection Select(int requestedMajor) {
    return selectAdapterLibrary(
        requestedMajor,
        [this](const char* name) -> void* {
          attempts_.emplace_back(name);
          auto library = libraries_.find(name);
          return library == libraries_.end() ? nullptr : &library->second;
        },
        [this](void* handle) {
          closed_.push_back(static_cast<FakeLibrary*>(handle)->name);
        },
        [](void* handle) { return static_cast<FakeLibrary*>(handle)->major; });
  }

  std::map<std::string, FakeLibrary> libraries_;
  std::vector<std::string> attempts_;
  std::vector<std::string> closed_;
};

TEST_F(
    AdapterLibrarySelectorTest,
    ExplicitMajorFallsBackToCompatibleUnversionedLibrary) {
  AddLibrary("libneuronusdk_adapter.mtk.so", 8);

  const auto selection = Select(8);

  EXPECT_NE(selection.handle, nullptr);
  EXPECT_EQ(selection.major, 8);
  EXPECT_EQ(
      attempts_,
      (std::vector<std::string>{
          "libneuronusdk_adapter.8.mtk.so", "libneuronusdk_adapter.mtk.so"}));
  EXPECT_TRUE(closed_.empty());
}

TEST_F(
    AdapterLibrarySelectorTest,
    ExplicitMajorClosesIncompatibleCandidateBeforeFallback) {
  AddLibrary("libneuronusdk_adapter.8.mtk.so", 9);
  AddLibrary("libneuronusdk_adapter.mtk.so", 8);

  const auto selection = Select(8);

  EXPECT_NE(selection.handle, nullptr);
  EXPECT_EQ(selection.major, 8);
  EXPECT_EQ(
      closed_, (std::vector<std::string>{"libneuronusdk_adapter.8.mtk.so"}));
}

TEST_F(AdapterLibrarySelectorTest, ExplicitMajorRejectsLibraryWithoutVersion) {
  AddLibrary("libneuronusdk_adapter.mtk.so", -1);

  const auto selection = Select(8);

  EXPECT_EQ(selection.handle, nullptr);
  EXPECT_EQ(selection.major, -1);
  EXPECT_EQ(
      closed_, (std::vector<std::string>{"libneuronusdk_adapter.mtk.so"}));
}

TEST_F(AdapterLibrarySelectorTest, ExplicitMajorRejectsAllIncompatibleAliases) {
  AddLibrary("libneuronusdk_adapter.8.mtk.so", 9);
  AddLibrary("libneuronusdk_adapter.mtk.so", 9);
  AddLibrary("libneuron_adapter_mgvi.so", 9);
  AddLibrary("libneuron_adapter.so", 9);

  const auto selection = Select(8);

  EXPECT_EQ(selection.handle, nullptr);
  EXPECT_EQ(selection.major, -1);
  EXPECT_EQ(
      attempts_,
      (std::vector<std::string>{
          "libneuronusdk_adapter.8.mtk.so",
          "libneuronusdk_adapter.mtk.so",
          "libneuron_adapter_mgvi.so",
          "libneuron_adapter.so"}));
  EXPECT_EQ(closed_, attempts_);
}

TEST_F(AdapterLibrarySelectorTest, LegacySelectionPreservesAliasOrder) {
  AddLibrary("libneuron_adapter.so", -1);

  const auto selection = Select(0);

  EXPECT_NE(selection.handle, nullptr);
  EXPECT_EQ(selection.major, 0);
  EXPECT_EQ(
      attempts_,
      (std::vector<std::string>{
          "libneuronusdk_adapter.mtk.so",
          "libneuron_adapter_mgvi.so",
          "libneuron_adapter.so"}));
}

TEST(AdapterLibraryMajorCompatibilityTest, NoSpecReusesSelectedLibrary) {
  EXPECT_TRUE(isAdapterLibraryMajorCompatible(9, 0));
  EXPECT_TRUE(isAdapterLibraryMajorCompatible(9, 9));
  EXPECT_FALSE(isAdapterLibraryMajorCompatible(9, 8));
}

TEST_F(
    AdapterLibrarySelectorTest,
    FailedConfigurationDoesNotPoisonLaterRequest) {
  AdapterLibraryState state;
  auto select = [this](int major) { return Select(major); };

  EXPECT_FALSE(configureAdapterLibraryState(state, 8, select));
  EXPECT_EQ(state.major, -1);
  EXPECT_FALSE(state.loadAttempted);

  AddLibrary("libneuronusdk_adapter.9.mtk.so", 9);
  EXPECT_TRUE(configureAdapterLibraryState(state, 9, select));
  EXPECT_NE(state.handle, nullptr);
  EXPECT_EQ(state.major, 9);
  EXPECT_TRUE(state.loadAttempted);
}

TEST_F(AdapterLibrarySelectorTest, SuccessfulConfigurationPinsOneMajor) {
  AdapterLibraryState state;
  auto select = [this](int major) { return Select(major); };
  AddLibrary("libneuronusdk_adapter.9.mtk.so", 9);

  EXPECT_TRUE(configureAdapterLibraryState(state, 9, select));
  attempts_.clear();

  EXPECT_TRUE(configureAdapterLibraryState(state, 0, select));
  EXPECT_TRUE(configureAdapterLibraryState(state, 9, select));
  EXPECT_FALSE(configureAdapterLibraryState(state, 8, select));
  EXPECT_TRUE(attempts_.empty());
}

TEST_F(AdapterLibrarySelectorTest, DefaultSelectionRecordsReportedMajor) {
  AdapterLibraryState state;
  auto select = [this](int major) { return Select(major); };
  AddLibrary("libneuronusdk_adapter.mtk.so", 9);

  EXPECT_NE(getOrSelectAdapterLibrary(state, select), nullptr);
  EXPECT_EQ(state.major, 9);
  attempts_.clear();

  EXPECT_TRUE(configureAdapterLibraryState(state, 9, select));
  EXPECT_TRUE(attempts_.empty());
}

TEST_F(AdapterLibrarySelectorTest, UnknownLegacyMajorRejectsExplicitRequest) {
  AdapterLibraryState state;
  auto select = [this](int major) { return Select(major); };
  AddLibrary("libneuron_adapter.so", -1);

  EXPECT_NE(getOrSelectAdapterLibrary(state, select), nullptr);
  EXPECT_EQ(state.major, 0);
  attempts_.clear();

  EXPECT_FALSE(configureAdapterLibraryState(state, 9, select));
  EXPECT_TRUE(attempts_.empty());
}

TEST_F(
    AdapterLibrarySelectorTest,
    FailedDefaultLookupKeepsSubsequentConfigurationFailClosed) {
  AdapterLibraryState state;
  auto select = [this](int major) { return Select(major); };

  EXPECT_EQ(getOrSelectAdapterLibrary(state, select), nullptr);
  EXPECT_TRUE(state.loadAttempted);
  EXPECT_EQ(state.major, 0);
  attempts_.clear();
  EXPECT_EQ(getOrSelectAdapterLibrary(state, select), nullptr);
  EXPECT_TRUE(attempts_.empty());

  AddLibrary("libneuronusdk_adapter.9.mtk.so", 9);
  EXPECT_FALSE(configureAdapterLibraryState(state, 9, select));
  EXPECT_FALSE(configureAdapterLibraryState(state, 0, select));
  EXPECT_EQ(state.handle, nullptr);
  EXPECT_EQ(state.major, 0);
}

TEST(AdapterLibraryMajorParserTest, AcceptsBoundedDecimalMajor) {
  EXPECT_EQ(parseAdapterLibraryMajor("9", 1), 9);
  EXPECT_EQ(parseAdapterLibraryMajor("09", 2), 9);
  EXPECT_EQ(parseAdapterLibraryMajor("255", 3), 255);
}

TEST(AdapterLibraryMajorParserTest, RejectsMalformedOrOutOfRangeMajor) {
  EXPECT_EQ(parseAdapterLibraryMajor(nullptr, 1), -1);
  EXPECT_EQ(parseAdapterLibraryMajor("", 0), -1);
  EXPECT_EQ(parseAdapterLibraryMajor("0", 1), -1);
  EXPECT_EQ(parseAdapterLibraryMajor("256", 3), -1);
  EXPECT_EQ(parseAdapterLibraryMajor("1234", 4), -1);
  EXPECT_EQ(parseAdapterLibraryMajor("9a", 2), -1);
}

TEST(AdapterLibraryMajorCompileSpecTest, ReturnsZeroWhenAbsent) {
  std::array<char, 1> value{'1'};
  std::array<runtime::CompileSpec, 1> compileSpecs{{
      {"unrelated", {value.data(), value.size()}},
  }};

  EXPECT_EQ(scanAdapterLibraryMajor(compileSpecs), 0);
}

TEST(AdapterLibraryMajorCompileSpecTest, ReturnsRequestedMajorWhenValid) {
  std::array<char, 1> value{'9'};
  std::array<runtime::CompileSpec, 1> compileSpecs{{
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
  }};

  EXPECT_EQ(scanAdapterLibraryMajor(compileSpecs), 9);
}

TEST(AdapterLibraryMajorCompileSpecTest, RejectsMalformedMajor) {
  std::array<char, 2> value{'9', 'a'};
  std::array<runtime::CompileSpec, 1> compileSpecs{{
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
  }};

  EXPECT_EQ(scanAdapterLibraryMajor(compileSpecs), -1);
}

TEST(AdapterLibraryMajorCompileSpecTest, RejectsDuplicateMajor) {
  std::array<char, 1> value{'9'};
  std::array<runtime::CompileSpec, 2> compileSpecs{{
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
  }};

  EXPECT_EQ(scanAdapterLibraryMajor(compileSpecs), -1);
}

} // namespace
} // namespace neuron
} // namespace backends
} // namespace executorch
