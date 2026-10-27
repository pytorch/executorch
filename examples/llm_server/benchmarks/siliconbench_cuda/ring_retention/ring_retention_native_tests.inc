// Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license in LICENSE.
#include <map>
namespace {
class CudaSplitRingCacheTest : public CudaIndependentSessionCacheTest {
 protected:
  static std::shared_ptr<cache::Cache> split() {
    auto cfg = config(kCapacity, 4, kMaxWrite);
    cfg.max_seqs = kMaxSeqs; cfg.runtime_sequence_columns = true;
    cfg.independent_sequence_capacity = true; cfg.split_ring_history = true;
    return cu::make_cuda_batched_sequence_kv_cache(flat_and_ring(2), cfg);
  }
  static FakeContainer split_container() {
    auto fake = independent_container();
    for (int i = 2; i < 4; ++i) declare(fake, fake.fqns[i], slimc10::ScalarType::BFloat16,
        {1, 2 * kDeclaredColumns, kHeads, kDim});
    fake.internal_names.push_back("ring_stage");
    fake.fqns.push_back("__et_offgraph_kv_ring_stage");
    declare(fake, fake.fqns.back(), slimc10::ScalarType::BFloat16,
        {1, 2, kMaxWrite, kHeads, kDim});
    return fake;
  }
  // Emulate the compiled commit (kernel numerics are covered separately):
  // populate packed staging and write only the final window to each owner.
  static Error write(cache::Cache& object, cu::CudaDelegateHandle& handle,
                     FakeContainer& fake, const std::vector<int32_t>& ids, int width=0) {
    auto* c=object.as<cache::BatchControl>();auto* kv=object.as<cu::CudaKVCache>();
    if (!c->declare_step(ids)) return Error::InvalidArgument;
    width=width ? width : ids.size();
    ET_CHECK_OK_OR_RETURN_ERROR(kv->prepare_step(width,cudaStreamPerThread));
    ET_CHECK_OK_OR_RETURN_ERROR(kv->rebind_for_execute(&handle));
    if(cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess)return Error::Internal;
    const auto m=info(fake);std::vector<uint16_t> staging(2*kMaxWrite*kRow,0);
    std::map<int,int> final;
    for(int s=0;s<kSegments;++s)if(m[s*7+1]) final[m[s*7+2]]=std::max(final[m[s*7+2]],m[s*7+3]);
    for(int s=0;s<kSegments;++s) {
      int st=m[s*7],n=m[s*7+1],col=m[s*7+2],end=m[s*7+3],pitch=m[s*7+4];
      for(int j=0;j<n;++j)for(int v=0;v<2;++v) {
        int pos=end-n+j;uint16_t value=1000*v+100*ids[st+j]+pos+1;
        std::fill(staging.begin()+(v*kMaxWrite+st+j)*kRow,
            staging.begin()+(v*kMaxWrite+st+j+1)*kRow,value);
        if(pos>=final[col]-2) {
          std::vector<uint16_t> row(kRow,value);
          auto* dst=static_cast<uint16_t*>(fake.bound[v ? "ring_v" : "ring_k"].data)+(pos%2*pitch+col)*kRow;
          if(cudaMemcpy(dst,row.data(),kRow*sizeof(uint16_t),cudaMemcpyHostToDevice)!=cudaSuccess)return Error::Internal;
        }
      }
    }
    if(cudaMemcpy(fake.bound["ring_stage"].data,staging.data(),staging.size()*sizeof(uint16_t),cudaMemcpyHostToDevice)!=cudaSuccess)return Error::Internal;
    return kv->commit_step(width);
  }
  static void check_last(FakeContainer& fake,int column,int columns,int owner,int keep) {
    for(int v=0;v<2;++v) {
      const auto values=read_column(fake.bound[v ? "ring_v" : "ring_k"].data,2,columns,column);
      const int pos=keep-1; // The next causal query needs window-1 old tokens.
      for(int d=0;d<kRow;++d)EXPECT_EQ(values[pos%2*kRow+d],1000*v+100*owner+pos+1);
    }
  }
};
}

TEST_F(CudaSplitRingCacheTest, PackedStagingKeepsMaximumRewindAndCloneContract) {
 auto object=split();ASSERT_TRUE(object);auto* c=object->as<cache::BatchControl>();auto* kv=object->as<cu::CudaKVCache>();
 auto fake=split_container();auto h=make_handle(fake);ASSERT_TRUE(kv->note_handle(&h).get());
 int a=*c->seq_new();ASSERT_EQ(write(*object,h,fake,std::vector<int32_t>(32,a)),Error::Ok);
 ASSERT_EQ(write(*object,h,fake,std::vector<int32_t>(32,a)),Error::Ok);
 ASSERT_TRUE(c->rewind(a,32));check_last(fake,0,kv->metrics().physical_columns,a,32);
 EXPECT_FALSE(c->rewind(a,31));
 auto clone=c->seq_clone(a,32);ASSERT_TRUE(clone);
 ASSERT_EQ(write(*object,h,fake,{*clone}),Error::Ok);
 auto m=info(fake);check_last(fake,m[2],m[4],*clone,33);
 ASSERT_TRUE(c->seq_rm(a));ASSERT_EQ(kv->compact_columns(0),Error::Ok);
 ASSERT_TRUE(c->rewind(*clone,32));check_last(fake,0,2,a,32);
 kv->forget_handle(&h);
}

TEST_F(CudaSplitRingCacheTest, RepeatedOwnerPackedRunsAndPaddedRowsRemainIsolated) {
 auto object=split();auto* c=object->as<cache::BatchControl>();auto* kv=object->as<cu::CudaKVCache>();
 auto fake=split_container();auto h=make_handle(fake);ASSERT_TRUE(kv->note_handle(&h).get());
 int a=*c->seq_new(),b=*c->seq_new();ASSERT_EQ(write(*object,h,fake,{a,a,b,a,a},32),Error::Ok);
 ASSERT_TRUE(c->rewind(a,2));check_last(fake,0,3,a,2);
 auto clone=c->seq_clone(b,1);ASSERT_TRUE(clone);
 ASSERT_EQ(write(*object,h,fake,{*clone}),Error::Ok);auto m=info(fake);
 check_last(fake,m[2],m[4],*clone,2);
 ASSERT_TRUE(c->seq_rm(a));int reused=*c->seq_new();EXPECT_EQ(a,reused);
 ASSERT_EQ(write(*object,h,fake,{reused}),Error::Ok);m=info(fake);check_last(fake,m[2],m[4],reused,1);
 kv->forget_handle(&h);
}

TEST_F(CudaSplitRingCacheTest, IdleTrimPreservesLiveBindingsAndGraphReplay) {
 auto object=split();auto* c=object->as<cache::BatchControl>();auto* kv=object->as<cu::CudaKVCache>();
 auto fake=split_container();auto h=make_handle(fake);ASSERT_TRUE(kv->note_handle(&h).get());
 int a=*c->seq_new();ASSERT_EQ(write(*object,h,fake,{a}),Error::Ok);
 const auto ring=fake.bound["ring_k"].data;const auto stage=fake.bound["ring_stage"].data;
 h.cuda_graph_state.phase=cu::CudaGraphPhase::Replay;
 ASSERT_EQ(kv->compact_columns(0),Error::Ok);
 ASSERT_EQ(write(*object,h,fake,{a}),Error::Ok);
 EXPECT_EQ(fake.bound["ring_k"].data,ring);EXPECT_EQ(fake.bound["ring_stage"].data,stage);
 EXPECT_EQ(h.cuda_graph_state.phase,cu::CudaGraphPhase::Replay);
 EXPECT_TRUE(c->rewind(a,1));check_last(fake,0,2,a,1);
 kv->forget_handle(&h);
}

TEST_F(CudaSplitRingCacheTest, CloneWritesDoNotOverwriteSourceUndoPages) {
 auto object=split();auto* c=object->as<cache::BatchControl>();auto* kv=object->as<cu::CudaKVCache>();
 auto fake=split_container();auto h=make_handle(fake);ASSERT_TRUE(kv->note_handle(&h).get());
 int a=*c->seq_new();ASSERT_EQ(write(*object,h,fake,std::vector<int32_t>(32,a)),Error::Ok);
 ASSERT_EQ(write(*object,h,fake,std::vector<int32_t>(16,a)),Error::Ok);
 auto clone=c->seq_clone(a,48);ASSERT_TRUE(clone);
 ASSERT_EQ(write(*object,h,fake,{*clone}),Error::Ok);
 ASSERT_TRUE(c->rewind(a,16));check_last(fake,0,3,a,16);
 ASSERT_TRUE(c->rewind(*clone,17));check_last(fake,1,3,a,17);
 kv->forget_handle(&h);
}
