/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <fstream>

#include <executorch/examples/qualcomm/oss_scripts/llama3_2/runner/io_memory.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

Memory::Memory(std::vector<std::shared_ptr<Module>>& modules)
    : data_ptr_(nullptr, [](void*) {}),
      input_tensors_(modules.size()),
      output_tensors_(modules.size()),
      modules_(modules) {}

Memory::~Memory() {}

void* Memory::get_mutable_ptr() {
  return data_ptr_.get();
}

std::vector<Tensor> Memory::get_input_tensors(int shard_index) {
  std::vector<Tensor> ret;
  ret.reserve(input_tensors_.size());
  for (TensorImpl* impl : input_tensors_[shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

std::vector<Tensor> Memory::get_output_tensors(int shard_index) {
  std::vector<Tensor> ret;
  ret.reserve(output_tensors_.size());
  for (TensorImpl* impl : output_tensors_[shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

KVCachedMemory::KVCachedMemory(std::vector<std::shared_ptr<Module>>& modules)
    : Memory(modules),
      shard_layers_({QNN_LLAMA3_2_NUM_LAYERS}),
      num_heads_(QNN_LLAMA3_2_NUM_HEADS) {
  data_ptr_ = std::unique_ptr<void, void (*)(void*)>(
      new IO, [](void* ptr) { delete static_cast<IO*>(ptr); });
}

void KVCachedMemory::prepare_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::memset(ptr, 0, sizeof(IO));
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  // [I]: input_tokens
  Result<TensorInfo> input_tok = methods_meta[0]->input_tensor_meta(0);
  input_tok_ = std::make_unique<TensorImpl>(
      input_tok->scalar_type(),
      input_tok->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_tok->sizes().data()),
      &ptr->input_tok,
      const_cast<TensorImpl::DimOrderType*>(input_tok->dim_order().data()));
  input_tensors_[0].push_back(input_tok_.get());

  // [I]: input_pos
  Result<TensorInfo> input_pos = methods_meta[0]->input_tensor_meta(1);
  input_pos_ = std::make_unique<TensorImpl>(
      input_pos->scalar_type(),
      input_pos->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
      &ptr->input_pos,
      const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
  input_tensors_[0].push_back(input_pos_.get());

  // [I]: atten_mask
  std::fill(
      ptr->attention_mask, ptr->attention_mask + QNN_LLAMA3_2_SEQLEN, -255);
  Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(2);
  attention_mask_ = std::make_unique<TensorImpl>(
      atten_mask->scalar_type(),
      atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
      ptr->attention_mask,
      const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
  input_tensors_[0].push_back(attention_mask_.get());

  // [I] kv_cache
  int index = 3; // bypass input_tokens, input_pos, atten_mask
  for (int offset = 0,
           shard_index = 0,
           v_stride = (QNN_LLAMA3_2_SEQLEN - 1) * QNN_LLAMA3_2_HEAD_DIM;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->input_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_in_ : v_cache_in_);
          void* cache_ptr = (cache_group == 0)
              ? static_cast<void*>(ptr->k_cache[layer + offset][head])
              : static_cast<void*>(
                    ptr->v_cache[layer + offset] + head * v_stride);

          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          input_tensors_[shard_index].push_back(cache.back().get());
        }
      }
    }
  }

  // [O]: logits
  int logit_index = 0;
  Result<TensorInfo> logits =
      methods_meta[modules_.size() - 1]->output_tensor_meta(logit_index);
  logits_ = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      ptr->logits,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[modules_.size() - 1].push_back(logits_.get());

  // [O] kv_cache
  index = 1;
  for (int offset = 0,
           shard_index = 0,
           v_stride = (QNN_LLAMA3_2_SEQLEN - 1) * QNN_LLAMA3_2_HEAD_DIM;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_out_ : v_cache_out_);
          void* cache_ptr = (cache_group == 0)
              ? static_cast<void*>(ptr->k_cache_out[layer + offset][head])
              : static_cast<void*>(
                    ptr->v_cache[layer + offset] + (head + 1) * v_stride);
          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          output_tensors_[shard_index].push_back(cache.back().get());
        }
      }
    }
  }
}

void KVCachedMemory::update_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  int seq_len = (QNN_LLAMA3_2_SEQLEN - 1);
  // update input_tok
  ptr->input_tok = static_cast<int32_t>(cur_token);
  // update position_ids
  ptr->input_pos = static_cast<int32_t>(pos);
  // update causal mask for next token
  ptr->attention_mask[seq_len - pos] = 0;

  // update v_cache
  for (int i = 0; i < v_cache_in_.size(); i++) {
    v_cache_in_[i]->set_data(
        v_cache_in_[i]->mutable_data<uint8_t>() + QNN_LLAMA3_2_HEAD_DIM);
    v_cache_out_[i]->set_data(
        v_cache_out_[i]->mutable_data<uint8_t>() + QNN_LLAMA3_2_HEAD_DIM);
  }
  for (int shard = 0; shard < output_tensors.size(); shard++) {
    for (int index = 0; index < output_tensors[shard].size(); index++) {
      ET_CHECK_MSG(
          modules_[shard]->set_output(output_tensors[shard][index], index) ==
              Error::Ok,
          "failed to set output tensor for module %d's %d'th output "
          "while updating kv_cache output tensors",
          shard,
          index);
    }
  }

  // update k_cache by single thread, this part is cpu cache sensitive
  for (int i = 0; i < k_cache_in_.size(); ++i) {
    uint8_t* ptr_in = k_cache_in_[i]->mutable_data<uint8_t>();
    const uint8_t* ptr_out = k_cache_out_[i]->data<uint8_t>();
    for (size_t j = 0, offset = seq_len; j < QNN_LLAMA3_2_HEAD_DIM;
         ++j, offset += seq_len) {
      ptr_in[offset] = ptr_out[j];
    }
    k_cache_in_[i]->set_data(ptr_in + 1);
  }
}

} // namespace example
