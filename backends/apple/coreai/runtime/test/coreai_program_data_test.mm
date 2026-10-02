/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_pte_fixture.h"

#include <executorch/backends/apple/coreai/runtime/coreai_pte.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/extended_header.h>
#include <gtest/gtest.h>

#include <cstring>
#include <utility>

namespace executorch::backends::coreai::testing {
namespace {
namespace fb = executorch_flatbuffer;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::ExtendedHeader;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Program;
using SegmentType = DataLoader::SegmentInfo::Type;
using LoadFault = FBSyntheticPTE::LoadFault;

class CoreAIProgramDataTest : public ::testing::Test {
 protected:
  void SetUp() override { executorch::runtime::runtime_init(); }
};

NSData* payload(const char* value = "abc") {
  return [NSData dataWithBytes:value length:std::strlen(value)];
}

template <typename T>
void set_field(const void* table, flatbuffers::voffset_t field, T value) {
  ASSERT_NE(table, nullptr);
  auto* fields = static_cast<flatbuffers::Table*>(const_cast<void*>(table));
  ASSERT_TRUE(fields->SetField<T>(field, value));
}

void expect_payload(const FreeableBuffer& buffer, NSData* expected) {
  ASSERT_EQ(buffer.size(), expected.length);
  ASSERT_NE(buffer.data(), nullptr);
  EXPECT_EQ(std::memcmp(buffer.data(), expected.bytes, expected.length), 0);
}

void expect_invalid(FBSyntheticPTE& pte, Error error = Error::InvalidProgram) {
  {
    auto result = inspect_coreai_pte(pte);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error(), error);
  }
  EXPECT_TRUE(pte.check_released());
}

TEST_F(CoreAIProgramDataTest, BorrowsInlineAndOwnsSegmentsAcrossMethods) {
  NSData* first = payload("abc");
  NSData* second = payload("xyz");
  FBSyntheticPTE pte({{"forward", {{"CoreAIBackend", first}, {"UnregisteredBackend", first, true}}},
                      {"other", {{"CoreAIBackend", second, true}}}});
  {
    auto result = inspect_coreai_pte(pte);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->processed.size(), 2);
    ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[0], first));
    ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[1], second));
    const auto* program = fb::GetProgram(result->program_storage.data());
    const auto* inline_data = program->backend_delegate_data()->Get(0)->data()->data();
    EXPECT_EQ(result->processed[0].data(), inline_data);
    EXPECT_NE(result->program_storage.data(), pte.bytes.data());
    ASSERT_EQ(pte.requests.size(), 3);
    EXPECT_EQ(result->program_storage.data(), pte.requests[1].data);
    EXPECT_EQ(result->processed[1].data(), pte.requests[2].data);
    EXPECT_EQ(pte.backend_indices, (std::vector<size_t>{1}));
    EXPECT_EQ(pte.backend_descriptors, (std::vector<std::string>{"CoreAIBackend"}));
    EXPECT_EQ(pte.program_loads, 2);
    EXPECT_EQ(pte.program_releases, 1);
    EXPECT_EQ(pte.backend_releases, 0);
    const int releases = pte.program_releases;
    result->processed[0].Free();
    EXPECT_EQ(pte.program_releases, releases);
    EXPECT_EQ(std::memcmp(inline_data, first.bytes, first.length), 0);
    FreeableBuffer moved(std::move(result->processed[1]));
    result->processed[1].Free();
    EXPECT_EQ(pte.backend_releases, 0);
    ASSERT_NO_FATAL_FAILURE(expect_payload(moved, second));
    moved.Free();
    moved.Free();
    EXPECT_EQ(pte.backend_releases, 1);
  }
  EXPECT_TRUE(pte.program_alive_at_backend_release);
  EXPECT_TRUE(pte.check_released());
}

TEST_F(CoreAIProgramDataTest, RoutesNonidentityGlobalReferencesInMethodOrder) {
  for (bool segmented : {false, true}) {
    SCOPED_TRACE(segmented ? "segmented" : "inline");
    auto row = [&] {
      NSData* first = payload("abc");
      NSData* second = payload("xyz");
      ASSERT_EQ(first.length, second.length);
      ASSERT_NE(std::memcmp(first.bytes, second.bytes, first.length), 0);
      FBSyntheticPTE pte(
          {{"forward", {{"CoreAIBackend", payload("---"), segmented, 2}}},
           {"other",
            {{"CoreAIBackend", second, segmented, 1}, {"UnregisteredBackend", first, segmented}}}});
      const auto* source = fb::GetProgram(pte.bytes.data());
      const auto header = ExtendedHeader::Parse(pte.bytes.data(), pte.bytes.size());
      ASSERT_TRUE(header.ok());
      {
        auto result = inspect_coreai_pte(pte);
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result->processed.size(), 2);
        ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[0], first));
        ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[1], second));
        ASSERT_GE(pte.requests.size(), 2);
        for (size_t i = 0; i < 2; ++i) {
          EXPECT_EQ(pte.requests[i].offset, 0);
          EXPECT_EQ(pte.requests[i].type, SegmentType::Program);
          EXPECT_EQ(pte.requests[i].index, 0);
          EXPECT_TRUE(pte.requests[i].descriptor.empty());
        }
        EXPECT_EQ(pte.requests[0].size, ExtendedHeader::kNumHeadBytes);
        EXPECT_EQ(pte.requests[1].size, header->program_size);
        if (segmented) {
          ASSERT_EQ(pte.requests.size(), 4);
          EXPECT_EQ(pte.backend_indices, (std::vector<size_t>{2, 1}));
          for (size_t i = 0; i < 2; ++i) {
            const size_t index = i == 0 ? 2 : 1;
            const auto* segment = source->segments()->Get(index);
            const auto& request = pte.requests[i + 2];
            EXPECT_EQ(request.type, SegmentType::Backend);
            EXPECT_EQ(request.index, index);
            EXPECT_EQ(request.descriptor, "CoreAIBackend");
            EXPECT_EQ(request.offset, header->segment_base_offset + segment->offset());
            EXPECT_EQ(request.size, segment->size());
            EXPECT_EQ(request.data, result->processed[i].data());
          }
        } else {
          ASSERT_EQ(pte.requests.size(), 2);
          const auto* program = fb::GetProgram(result->program_storage.data());
          EXPECT_EQ(result->processed[0].data(),
                    program->backend_delegate_data()->Get(2)->data()->data());
          EXPECT_EQ(result->processed[1].data(),
                    program->backend_delegate_data()->Get(1)->data()->data());
        }
      }
      EXPECT_TRUE(pte.check_released());
      EXPECT_TRUE(pte.program_alive_at_backend_release);
    };
    ASSERT_NO_FATAL_FAILURE(row());
  }
}

TEST_F(CoreAIProgramDataTest, MatchesCompleteBackendIdentifier) {
  const char embedded[] = "CoreAIBackend\0suffix";
  CacheDelegateSpec nul_id{embedded, payload(), true};
  nul_id.backend_length = sizeof(embedded) - 1;
  nul_id.missing_processed = true;
  FBSyntheticPTE pte({{"forward",
                       {{"CoreAIBackendSuffix", payload(), true, 999},
                        {"CoreAIBacken", payload(), true, 999},
                        {"", payload(), true, 999},
                        nul_id,
                        {"CoreAIBackend", payload("yes")}}}});
  {
    auto result = inspect_coreai_pte(pte);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->processed.size(), 1);
    ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[0], payload("yes")));
    EXPECT_EQ(pte.backend_loads, 0);
    EXPECT_EQ(pte.requests.size(), 2);
  }
  EXPECT_TRUE(pte.check_released());
}

TEST_F(CoreAIProgramDataTest, AcceptsEmptyAndUnselectedPrograms) {
  for (bool other_backend : {false, true}) {
    SCOPED_TRACE(other_backend ? "no CoreAI delegate" : "no plans");
    std::vector<CacheMethodSpec> methods;
    if (other_backend) {
      methods.push_back({"forward", {{"UnregisteredBackend", payload(), true, 999}}});
    }
    FBSyntheticPTE pte(methods);
    {
      auto result = inspect_coreai_pte(pte);
      ASSERT_TRUE(result.ok());
      EXPECT_TRUE(result->processed.empty());
      EXPECT_EQ(pte.backend_loads, 0);
    }
    EXPECT_TRUE(pte.check_released());
  }
}

TEST_F(CoreAIProgramDataTest, AcceptsHeaderlessAndExtendedPrograms) {
  for (bool extended : {false, true}) {
    SCOPED_TRACE(extended ? "extended" : "headerless");
    FBSyntheticPTE pte({{"forward", {{"CoreAIBackend", payload(), extended}}}}, extended);
    {
      auto result = inspect_coreai_pte(pte);
      ASSERT_TRUE(result.ok());
      ASSERT_EQ(result->processed.size(), 1);
      ASSERT_NO_FATAL_FAILURE(expect_payload(result->processed[0], payload()));
      ASSERT_GE(pte.requests.size(), 2);
      EXPECT_EQ(pte.requests[0].size, ExtendedHeader::kNumHeadBytes);
      if (!extended) EXPECT_EQ(pte.requests[1].size, pte.bytes.size());
    }
    EXPECT_TRUE(pte.check_released());
  }
}

TEST_F(CoreAIProgramDataTest, RejectsMalformedFlatbufferAndUnsupportedVersion) {
  for (bool version : {false, true}) {
    SCOPED_TRACE(version ? "unsupported version" : "bad identifier");
    FBSyntheticPTE pte({{"forward", {{"CoreAIBackend", payload()}}}}, false);
    if (version) {
      ASSERT_NO_FATAL_FAILURE(set_field<uint32_t>(fb::GetProgram(pte.bytes.data()),
                                                  fb::Program::VT_VERSION,
                                                  Program::kMaxSupportedSchemaVersion + 1));
    } else {
      pte.bytes[4] = 'X';
    }
    ASSERT_NO_FATAL_FAILURE(expect_invalid(pte));
    EXPECT_EQ(pte.backend_loads, 0);
  }
}

TEST_F(CoreAIProgramDataTest, PropagatesLoadErrorsFromEveryPhase) {
  const char* phases[] = {"size", "prefix", "retained program", "selected segment"};
  for (int phase = 0; phase < 4; ++phase) {
    SCOPED_TRACE(phases[phase]);
    FBSyntheticPTE pte({{"forward", {{"CoreAIBackend", payload(), true}}}});
    if (phase == 0) {
      pte.size_error = Error::AccessFailed;
    } else {
      pte.fault_call = phase - 1;
      pte.load_fault = LoadFault::Error;
      pte.load_error = Error::AccessFailed;
    }
    ASSERT_NO_FATAL_FAILURE(expect_invalid(pte, Error::AccessFailed));
    EXPECT_EQ(pte.requests.size(), phase);
  }
}

TEST_F(CoreAIProgramDataTest, ReleasesEarlierBuffersOnLateFailure) {
  FBSyntheticPTE pte({{"forward",
                       {{"CoreAIBackend", payload()},
                        {"CoreAIBackend", payload("one"), true},
                        {"CoreAIBackend", payload("two"), true}}}});
  pte.fault_call = 3;
  pte.load_fault = LoadFault::Error;
  ASSERT_NO_FATAL_FAILURE(expect_invalid(pte, Error::NotSupported));
  EXPECT_GE(pte.backend_loads, 1);
  EXPECT_TRUE(pte.program_alive_at_backend_release);
  ASSERT_FALSE(pte.release_order.empty());
  EXPECT_EQ(pte.release_order.back(), SegmentType::Program);
}

}  // namespace
}  // namespace executorch::backends::coreai::testing
