// Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license in LICENSE.
namespace {
class CudaSplitRingBindingTest : public CudaSplitRingCacheTest {
 protected:
  static void omit(FakeContainer& fake, const std::string& fqn) {
    const auto it = std::find(fake.fqns.begin(), fake.fqns.end(), fqn);
    ASSERT_NE(it, fake.fqns.end());
    const auto index = std::distance(fake.fqns.begin(), it);
    fake.fqns.erase(it);
    fake.internal_names.erase(fake.internal_names.begin() + index);
  }
};
}

TEST_F(CudaSplitRingBindingTest, MissingUnusedColumnsLoadsAndCommits) {
  auto object = split();
  ASSERT_TRUE(object);
  auto* kv = object->as<cu::CudaKVCache>();
  auto* control = object->as<cache::BatchControl>();
  auto fake = split_container();
  omit(fake, "__et_offgraph_kv_seq_cols");
  auto handle = make_handle(fake);
  auto noted = kv->note_handle(&handle);
  ASSERT_TRUE(noted.ok());
  ASSERT_TRUE(*noted);
  const auto owner = control->seq_new();
  ASSERT_TRUE(owner);
  ASSERT_EQ(write(*object, handle, fake, {*owner}), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(CudaSplitRingBindingTest, MissingRequiredJournalIsRejected) {
  auto object = split();
  ASSERT_TRUE(object);
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = split_container();
  omit(fake, "__et_offgraph_kv_ring_stage");
  auto handle = make_handle(fake);
  EXPECT_EQ(kv->note_handle(&handle).error(), Error::InvalidProgram);
}

TEST_F(CudaSplitRingBindingTest, PresentOptionalColumnShapeIsValidated) {
  auto object = split();
  ASSERT_TRUE(object);
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = split_container();
  declare(fake, "__et_offgraph_kv_seq_cols", slimc10::ScalarType::Long, {kMaxWrite + 1});
  auto handle = make_handle(fake);
  EXPECT_EQ(kv->note_handle(&handle).error(), Error::InvalidProgram);
}
