// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the BSD-style license in LICENSE.

namespace {
class CudaIndependentSessionCacheTest : public CudaRuntimeColumnCacheTest {
 protected:
  static std::shared_ptr<cache::Cache> independent() {
    auto cfg = config(kCapacity, 4, kMaxWrite);
    cfg.max_seqs = kMaxSeqs;
    cfg.runtime_sequence_columns = true;
    cfg.independent_sequence_capacity = true;
    return cu::make_cuda_batched_sequence_kv_cache(flat_and_ring(2), cfg);
  }
  static FakeContainer independent_container() {
    auto fake = container(7);
    fake.internal_names.push_back("seq_base");
    fake.fqns.push_back("__et_offgraph_kv_seq_base");
    declare(fake, fake.fqns.back(), slimc10::ScalarType::Long, {kMaxWrite});
    return fake;
  }
  static std::vector<int32_t> info(FakeContainer& fake) {
    std::vector<int32_t> values(kSegments * 7);
    EXPECT_EQ(cudaMemcpy(values.data(), fake.bound["seq_info"].data,
        values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    return values;
  }
  static std::vector<uint16_t> read_flat(FakeContainer& fake, int offset, int rows) {
    std::vector<uint16_t> values(rows * kRow);
    EXPECT_EQ(cudaMemcpy(values.data(),
        static_cast<uint16_t*>(fake.bound["flat_k"].data) + offset * kRow,
        values.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), cudaSuccess);
    return values;
  }
  static void seed_flat(FakeContainer& fake, int offset, int rows, uint16_t value) {
    std::vector<uint16_t> values(rows * kRow, value);
    ASSERT_EQ(cudaMemcpy(
        static_cast<uint16_t*>(fake.bound["flat_k"].data) + offset * kRow,
        values.data(), values.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
  }
};
}

TEST_F(CudaIndependentSessionCacheTest, LongOwnerDoesNotGrowShortOwnerCapacity) {
  auto object = independent();
  auto* control = object->as<cache::BatchControl>();
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = independent_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int a = *control->seq_new(), b = *control->seq_new();
  ASSERT_EQ(step(*object, handle, {a,a,a,a,a,a,b}), Error::Ok);
  auto metadata = info(fake);
  EXPECT_EQ(metadata[6], 8);
  EXPECT_EQ(metadata[7 + 6], 4);
  seed_flat(fake, metadata[5], 6, 123);
  seed_flat(fake, metadata[7 + 5], 1, 456);
  const int64_t before = kv->metrics().allocated_bytes;
  ASSERT_EQ(step(*object, handle, {a,a,a,b}), Error::Ok);
  metadata = info(fake);
  EXPECT_EQ(metadata[6], 16);
  EXPECT_EQ(metadata[7 + 6], 4);
  EXPECT_EQ(read_flat(fake, metadata[5], 6), std::vector<uint16_t>(6*kRow,123));
  EXPECT_EQ(read_flat(fake, metadata[7+5],1), std::vector<uint16_t>(kRow,456));
  EXPECT_GT(kv->metrics().allocated_bytes, before);
  EXPECT_EQ(control->pos(a),9);
  EXPECT_EQ(control->pos(b),2);
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, CloneCompactionAndReusedIdPreserveIndependentOwnership) {
  auto object = independent();
  auto* control = object->as<cache::BatchControl>();
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = independent_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int a = *control->seq_new(), b = *control->seq_new();
  ASSERT_EQ(step(*object, handle, {a,a,a,a,a,a,b}), Error::Ok);
  auto metadata = info(fake);
  seed_flat(fake, metadata[5], 6,123);
  seed_flat(fake, metadata[7+5], 1,456);
  const auto clone = control->seq_clone(a,5);
  ASSERT_TRUE(clone.has_value());
  ASSERT_EQ(step(*object, handle, {*clone}), Error::Ok);
  metadata = info(fake);
  EXPECT_EQ(metadata[6],8);
  EXPECT_EQ(read_flat(fake,metadata[5],5),std::vector<uint16_t>(5*kRow,123));
  EXPECT_TRUE(control->seq_rm(a));
  EXPECT_TRUE(control->seq_rm(b));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(step(*object,handle,{*clone}),Error::Ok);
  metadata=info(fake);
  EXPECT_EQ(metadata[2],0);
  EXPECT_EQ(metadata[4],2);
  EXPECT_EQ(read_flat(fake,metadata[5],5),std::vector<uint16_t>(5*kRow,123));
  const int reused=*control->seq_new();
  EXPECT_EQ(reused,a);
  ASSERT_EQ(step(*object,handle,{reused}),Error::Ok);
  EXPECT_EQ(info(fake)[6],4);
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, SparseSixteenOwnersContractAndPaddingUsesDummyRange) {
  auto object=independent();
  auto* control=object->as<cache::BatchControl>();
  auto* kv=object->as<cu::CudaKVCache>();
  auto fake=independent_container();
  auto handle=make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> ids;
  for(int i=0;i<16;++i)ids.push_back(*control->seq_new());
  ASSERT_EQ(step(*object,handle,ids,32),Error::Ok);
  auto metadata=info(fake);
  EXPECT_EQ(metadata[4],17);
  for(int i=0;i<16;++i)EXPECT_EQ(metadata[i*7+6],4);
  std::vector<int64_t> bases(32);
  ASSERT_EQ(cudaMemcpy(bases.data(),fake.bound["seq_base"].data,
      bases.size()*sizeof(int64_t),cudaMemcpyDeviceToHost),cudaSuccess);
  for(int i=16;i<32;++i)EXPECT_EQ(bases[i],64);
  const int64_t before=kv->metrics().allocated_bytes;
  for(int i=0;i<16;++i)if(i!=5)ASSERT_TRUE(control->seq_rm(ids[i]));
  ASSERT_EQ(step(*object,handle,{ids[5]}),Error::Ok);
  EXPECT_EQ(info(fake)[4],2);
  EXPECT_LT(kv->metrics().allocated_bytes,before);
  control->clear();
  EXPECT_EQ(kv->metrics().live_sequences,0);
  EXPECT_EQ(kv->metrics().physical_columns,2);
  const int fresh = *control->seq_new();
  ASSERT_EQ(step(*object,handle,{fresh}),Error::Ok);
  EXPECT_EQ(info(fake)[6],4);
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, StableDecodeKeepsBindingsAndReplayUntilOwnerGrowth) {
  auto object=independent();
  auto* control=object->as<cache::BatchControl>();
  auto* kv=object->as<cu::CudaKVCache>();
  auto fake=independent_container();
  auto handle=make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int a=*control->seq_new();
  ASSERT_EQ(step(*object,handle,{a}),Error::Ok);
  handle.cuda_graph_state.phase=cu::CudaGraphPhase::Replay;
  const int updates=fake.updates;
  ASSERT_EQ(step(*object,handle,{a}),Error::Ok);
  EXPECT_EQ(fake.updates,updates);
  EXPECT_EQ(handle.cuda_graph_state.phase,cu::CudaGraphPhase::Replay);
  ASSERT_EQ(step(*object,handle,{a,a,a}),Error::Ok);
  EXPECT_GT(fake.updates,updates);
  EXPECT_NE(handle.cuda_graph_state.phase,cu::CudaGraphPhase::Replay);
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, RewindAndCloneIntoFreedSlotKeepRetainedPrefix) {
  auto object = independent();
  auto* control = object->as<cache::BatchControl>();
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = independent_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int a = *control->seq_new(), b = *control->seq_new();
  ASSERT_EQ(step(*object, handle, {a,a,a,a,a,a,b}), Error::Ok);
  const auto metadata = info(fake);
  seed_flat(fake, metadata[5], 6, 123);
  ASSERT_TRUE(control->seq_rm(b));
  const auto clone = control->seq_clone(a, 5);
  ASSERT_TRUE(clone.has_value());
  EXPECT_EQ(*clone, b);
  ASSERT_TRUE(control->rewind(*clone, 3));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(step(*object, handle, {*clone}), Error::Ok);
  const auto copied = info(fake);
  EXPECT_EQ(copied[6], 4);
  EXPECT_EQ(read_flat(fake, copied[5], 3), std::vector<uint16_t>(3*kRow,123));
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, CapturedCopyReplaysAfterOwnerGrowthAndRecapture) {
  auto object = independent();
  auto* control = object->as<cache::BatchControl>();
  auto* kv = object->as<cu::CudaKVCache>();
  auto fake = independent_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int a = *control->seq_new();
  ASSERT_EQ(step(*object, handle, {a}), Error::Ok);
  seed_flat(fake, info(fake)[5], 1, 321);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  void* output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, kRow*sizeof(uint16_t)), cudaSuccess);
  auto capture = [&]() {
    auto& graph = handle.cuda_graph_state;
    const auto base = info(fake)[5];
    EXPECT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
    EXPECT_EQ(cudaMemcpyAsync(output,
        static_cast<uint16_t*>(fake.bound["flat_k"].data) + base*kRow,
        kRow*sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamEndCapture(stream, &graph.graph), cudaSuccess);
    EXPECT_EQ(cudaGraphInstantiate(&graph.graph_exec, graph.graph, 0), cudaSuccess);
    graph.phase = cu::CudaGraphPhase::Replay;
  };
  auto replay = [&]() {
    EXPECT_EQ(cudaGraphLaunch(handle.cuda_graph_state.graph_exec, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<uint16_t> values(kRow);
    EXPECT_EQ(cudaMemcpy(values.data(), output, values.size()*sizeof(uint16_t),
        cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(values, std::vector<uint16_t>(kRow,321));
  };
  capture();
  replay();
  ASSERT_EQ(step(*object, handle, {a}), Error::Ok);
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  replay();
  ASSERT_EQ(step(*object, handle, {a,a,a}), Error::Ok);
  EXPECT_EQ(handle.cuda_graph_state.graph_exec, nullptr);
  capture();
  replay();
  handle.cuda_graph_state.release();
  ASSERT_EQ(cudaFree(output), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  kv->forget_handle(&handle);
}

TEST_F(CudaIndependentSessionCacheTest, RejectIndependentCapacityWithoutRuntimeColumns) {
  auto cfg = config(kCapacity, 4, kMaxWrite);
  cfg.max_seqs = kMaxSeqs;
  cfg.independent_sequence_capacity = true;
  EXPECT_EQ(cu::make_cuda_batched_sequence_kv_cache(flat_and_ring(2), cfg), nullptr);
}
