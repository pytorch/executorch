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

HybridMemory::HybridMemory(
    std::vector<std::shared_ptr<Module>>& modules,
    int32_t max_seq_len,
    int32_t vocab_size,
    int32_t num_layers,
    int32_t head_dim,
    int32_t num_heads)
    : Memory(modules),
      shard_layers_({num_layers}),
      max_seq_len_(max_seq_len),
      vocab_size_(vocab_size),
      num_layers_(num_layers),
      head_dim_(head_dim),
      num_heads_(num_heads) {
  data_ptr_ = std::unique_ptr<void, void (*)(void*)>(
      new IO, [](void* ptr) { delete static_cast<IO*>(ptr); });
}

void HybridMemory::prepare_kv_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::memset(ptr, 0, sizeof(IO));
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  // Init IO vector shape
  // atten_mask
  ptr->logits.resize(vocab_size_);
  ptr->attention_mask.resize(
      max_seq_len_, -255); // attention mask shape should be [1, ctx_length]
  // kv
  int32_t k_in_size = (head_dim_ + 1) * (max_seq_len_ - 1);
  int32_t k_out_size = num_heads_ * head_dim_;
  int32_t v_cache_size = (num_heads_ + 1) * (max_seq_len_ - 1) * head_dim_;
  for (int layer = 0; layer < num_layers_; layer++) {
    ptr->k_cache.emplace_back();
    for (int head = 0; head < num_heads_; head++) {
      ptr->k_cache[layer].emplace_back(std::vector<uint8_t>(k_in_size));
    }
    ptr->k_cache_out.emplace_back(std::vector<uint8_t>(k_out_size));
    ptr->v_cache.emplace_back(std::vector<uint8_t>(v_cache_size));
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

  // [I]: atten_mask
  Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(1);
  attention_mask_ = std::make_unique<TensorImpl>(
      atten_mask->scalar_type(),
      atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
      ptr->attention_mask.data(),
      const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
  input_tensors_[0].push_back(attention_mask_.get());

  // [I]: input_pos
  Result<TensorInfo> input_pos = methods_meta[0]->input_tensor_meta(2);
  input_pos_ = std::make_unique<TensorImpl>(
      input_pos->scalar_type(),
      input_pos->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
      &ptr->input_pos,
      const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
  input_tensors_[0].push_back(input_pos_.get());

  // [I] kv_cache
  int index = 3; // bypass input_tokens, input_pos, atten_mask
  for (int offset = 0,
           shard_index = 0,
           v_stride = (max_seq_len_ - 1) * head_dim_;
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
              ? static_cast<void*>(ptr->k_cache[layer + offset][head].data())
              : static_cast<void*>(
                    ptr->v_cache[layer + offset].data() + head * v_stride);

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
      ptr->logits.data(),
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[modules_.size() - 1].push_back(logits_.get());

  // [O] kv_cache
  index = 1;
  // Iterate through all kv cache outputs.
  // For k, we store it in k_cache_out and update to k_cache later.
  // For v, we append the output to the end of v_cache,
  // which serves as both input and output.
  for (int offset = 0,
           shard_index = 0,
           v_stride = (max_seq_len_ - 1) * head_dim_;
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
              ? static_cast<void*>(
                    ptr->k_cache_out[layer + offset].data() +
                    (head * head_dim_))
              : static_cast<void*>(
                    ptr->v_cache[layer + offset].data() +
                    (head + 1) * v_stride);
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

void HybridMemory::prepare_prefill_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::memset(ptr, 0, sizeof(IO));
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  // Parse some IO info from method meta
  // cache_len should be max_seq_len - 1
  int cache_len = methods_meta[0]->input_tensor_meta(0)->sizes()[1];

  // TODO: Combine vector init with KV mode once Hybrid mode is enabled
  // as it shares some common data structure.
  // Init IO vector shape
  ptr->prefill_input_toks.resize(cache_len);
  ptr->prefill_atten_mask.resize(cache_len * cache_len);
  ptr->prefill_logits.resize(cache_len * vocab_size_);
  // Init kv vector shape
  int32_t k_cache_out_size = num_heads_ * head_dim_ * cache_len;
  int32_t v_cache_size = (num_heads_ + 1) * cache_len * head_dim_;
  for (int layer = 0; layer < num_layers_; layer++) {
    ptr->k_cache_out.emplace_back(std::vector<uint8_t>(k_cache_out_size));
    ptr->v_cache.emplace_back(std::vector<uint8_t>(v_cache_size));
  }

  // [I]: pre_input_tokens
  Result<TensorInfo> prefill_input_toks = methods_meta[0]->input_tensor_meta(0);
  prefill_input_toks_ = std::make_unique<TensorImpl>(
      prefill_input_toks->scalar_type(),
      prefill_input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_input_toks->sizes().data()),
      ptr->prefill_input_toks.data(),
      const_cast<TensorImpl::DimOrderType*>(
          prefill_input_toks->dim_order().data()));
  input_tensors_[0].push_back(prefill_input_toks_.get());
  // [I]: prefill_attn_mask
  for (int i = 0; i < cache_len; ++i) {
    for (int j = 0; j < cache_len; ++j) {
      if (i < j) {
        ptr->prefill_atten_mask[i * cache_len + j] = -255;
      } else {
        ptr->prefill_atten_mask[i * cache_len + j] = 0;
      }
    }
  }

  Result<TensorInfo> prefill_attn_mask = methods_meta[0]->input_tensor_meta(1);
  prefill_attn_mask_ = std::make_unique<TensorImpl>(
      prefill_attn_mask->scalar_type(),
      prefill_attn_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_attn_mask->sizes().data()),
      ptr->prefill_atten_mask.data(),
      const_cast<TensorImpl::DimOrderType*>(
          prefill_attn_mask->dim_order().data()));
  input_tensors_[0].push_back(prefill_attn_mask_.get());

  // [O]: logits
  int logit_index = 0;
  Result<TensorInfo> logits =
      methods_meta[modules_.size() - 1]->output_tensor_meta(logit_index);
  logits_ = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      ptr->prefill_logits.data(),
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[modules_.size() - 1].push_back(logits_.get());
  // [O] kv_cache
  int index = 1;
  for (int offset = 0, shard_index = 0, cache_stride = cache_len * head_dim_;
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
              ? static_cast<void*>(
                    ptr->k_cache_out[layer + offset].data() +
                    head * cache_stride)
              : static_cast<void*>(
                    ptr->v_cache[layer + offset].data() + head * cache_stride);
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

void HybridMemory::update_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  int seq_len = (max_seq_len_ - 1);
  // update input_tok
  ptr->input_tok = static_cast<int32_t>(cur_token);
  // update position_ids
  ptr->input_pos = static_cast<int32_t>(pos);
  // update causal mask for next token
  ptr->attention_mask[seq_len - pos] = 0;

  // update v_cache
  for (int i = 0; i < v_cache_in_.size(); i++) {
    v_cache_in_[i]->set_data(
        v_cache_in_[i]->mutable_data<uint8_t>() + head_dim_);
    v_cache_out_[i]->set_data(
        v_cache_out_[i]->mutable_data<uint8_t>() + head_dim_);
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
    for (size_t j = 0, offset = seq_len; j < head_dim_;
         ++j, offset += seq_len) {
      ptr_in[offset] = ptr_out[j];
    }
    k_cache_in_[i]->set_data(ptr_in + 1);
  }
}

} // namespace example
