/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <fstream>

#include <executorch/examples/qualcomm/qaihub_scripts/llama/runner/io_memory.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

Memory::Memory(
    const std::vector<std::string>& pos_embs_path,
    std::vector<std::shared_ptr<Module>>& modules)
    : data_ptr_(nullptr, [](void*) {}),
      input_tensors_(modules.size()),
      output_tensors_(modules.size()),
      pos_embs_path_(pos_embs_path),
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

BertMemory::BertMemory(
    const std::vector<std::string>& pos_embs_path,
    std::vector<std::shared_ptr<Module>>& modules,
    std::vector<int> shard_layers)
    : Memory(pos_embs_path, modules),
      shard_layers_(shard_layers),
      num_heads_(QAIHUB_LLAMA_NUM_HEADS) {
  data_ptr_ = std::unique_ptr<void, void (*)(void*)>(
      new IO, [](void* ptr) { delete static_cast<IO*>(ptr); });
}

void BertMemory::prepare_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::memset(ptr, 0, sizeof(IO));

  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }
  // [I] position embedding initialization
  for (size_t i = 0; i < pos_embs_path_.size(); ++i) {
    std::ifstream fin(pos_embs_path_[i], std::ios::binary);
    fin.read(
        reinterpret_cast<char*>(
            i == 0 ? ptr->position_ids_cos : ptr->position_ids_sin),
        1024 * 64 * 2);
    fin.close();
  }
  // [I]: all shards (4 shards for llama2, 5 shards for llama)
  {
    // [I]: input_ids
    Result<TensorInfo> input_ids = methods_meta[0]->input_tensor_meta(0);
    input_ids_ = std::make_unique<TensorImpl>(
        input_ids->scalar_type(),
        input_ids->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_ids->sizes().data()),
        ptr->input_ids,
        const_cast<TensorImpl::DimOrderType*>(input_ids->dim_order().data()));
    input_tensors_[0].push_back(input_ids_.get());
    // [I]: atten_mask
    Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(1);
    attention_mask_ = std::make_unique<TensorImpl>(
        atten_mask->scalar_type(),
        atten_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
        ptr->attention_mask,
        const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
    input_tensors_[0].push_back(attention_mask_.get());
    // [I]: pos_ids_cos
    Result<TensorInfo> pos_ids_cos = methods_meta[0]->input_tensor_meta(2);
    position_ids_cos_ = std::make_unique<TensorImpl>(
        pos_ids_cos->scalar_type(),
        pos_ids_cos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(pos_ids_cos->sizes().data()),
        ptr->position_ids_cos,
        const_cast<TensorImpl::DimOrderType*>(pos_ids_cos->dim_order().data()));
    input_tensors_[0].push_back(position_ids_cos_.get());
    // [I]: pos_ids_sin
    Result<TensorInfo> pos_ids_sin = methods_meta[0]->input_tensor_meta(3);
    position_ids_sin_ = std::make_unique<TensorImpl>(
        pos_ids_sin->scalar_type(),
        pos_ids_sin->sizes().size(),
        const_cast<TensorImpl::SizesType*>(pos_ids_sin->sizes().data()),
        ptr->position_ids_sin,
        const_cast<TensorImpl::DimOrderType*>(pos_ids_sin->dim_order().data()));
    input_tensors_[0].push_back(position_ids_sin_.get());
    // [IO]: hidden_state => [I] shard2,3,4
    int output_index =
        shard_layers_[0] * 2 * num_heads_; // layers*(k + v caches)*heads
    Result<TensorInfo> hidden_state =
        methods_meta[0]->output_tensor_meta(output_index);
    hidden_state_ = std::make_unique<TensorImpl>(
        hidden_state->scalar_type(),
        hidden_state->sizes().size(),
        const_cast<TensorImpl::SizesType*>(hidden_state->sizes().data()),
        ptr->hidden_state,
        const_cast<TensorImpl::DimOrderType*>(
            hidden_state->dim_order().data()));
    // reuse inputs for following tensors
    for (int shard_index = 1; shard_index < modules_.size(); ++shard_index) {
      // inputs of shards 1 to n: hidden_state, atten_mask, pos_ids_cos,
      // pos_ids_sin
      input_tensors_[shard_index].push_back(hidden_state_.get());
      input_tensors_[shard_index].push_back(attention_mask_.get());
      input_tensors_[shard_index].push_back(position_ids_cos_.get());
      input_tensors_[shard_index].push_back(position_ids_sin_.get());
    }
  }
  // [O] kv_cache for all shards (4 shards for llama2 and 5 shards for llama3)
  for (int offset = 0, shard_index = 0; shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
      for (int cache_group = 0; cache_group < 2; ++cache_group) {
        for (int head = 0; head < num_heads_; ++head) {
          int index = num_heads_ * 2 * layer + cache_group * num_heads_ + head;
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? v_cache_ : k_cache_);
          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_group == 0 ? ptr->v_cache[layer + offset][head]
                               : ptr->k_cache[layer + offset][head],
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          output_tensors_[shard_index].push_back(cache.back().get());
        }
      }
    }
  }
  // [O]: hidden_state for shard 0 to n-1
  for (int shard_index = 0; shard_index < modules_.size() - 1; ++shard_index) {
    output_tensors_[shard_index].push_back(hidden_state_.get());
  }
  // [O]: logits
  {
    int output_index = shard_layers_[modules_.size() - 1] * 2 *
        num_heads_; // layers*(k + v caches)*heads
    Result<TensorInfo> logits =
        methods_meta[modules_.size() - 1]->output_tensor_meta(output_index);
    logits_ = std::make_unique<TensorImpl>(
        logits->scalar_type(),
        logits->sizes().size(),
        const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
        ptr->logits,
        const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
    output_tensors_[modules_.size() - 1].push_back(logits_.get());
  }
}

void BertMemory::update_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  (void)output_tensors;
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  static int num_tokens_generated = 0;
  int seq_len = 1024, last_index = seq_len - 1;
  // refill past token ids, which is equivalent to following snippet:
  // --->
  // for (int i = 0; i < last_index; ++i) {
  //   ptr->input_ids[i] = ptr->input_ids[i + 1];
  // }
  // ptr->input_ids[last_index] = static_cast<int32_t>(cur_token);
  // <---
  int32_t* new_addr = ++num_tokens_generated + ptr->input_ids;
  new_addr[last_index] = static_cast<int32_t>(cur_token);
  input_ids_->set_data(new_addr);
  // update causal mask for next token
  int tokens = pos + 1, start = last_index - tokens;
  for (int i = last_index; tokens >= 0; --i, --tokens) {
    ptr->attention_mask[i * seq_len + start] = 65535;
  }
}

KVCachedMemory::KVCachedMemory(
    const std::vector<std::string>& pos_embs_path,
    std::vector<std::shared_ptr<Module>>& modules,
    std::vector<int> shard_layers)
    : Memory(pos_embs_path, modules),
      shard_layers_(shard_layers),
      num_heads_(QAIHUB_LLAMA_NUM_HEADS) {
  data_ptr_ = std::unique_ptr<void, void (*)(void*)>(
      new IO, [](void* ptr) { delete static_cast<IO*>(ptr); });
  if (num_heads_ == 32) {
    futures_ = std::vector<std::future<void>>(thread_pool_.num_workers());
  }
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
  // [I] position embedding initialization
  for (size_t i = 0; i < pos_embs_path_.size(); ++i) {
    std::ifstream fin(pos_embs_path_[i], std::ios::binary);
    fin.read(
        reinterpret_cast<char*>(
            i == 0 ? ptr->position_ids_cos : ptr->position_ids_sin),
        1024 * 64 * 2);
    fin.close();
  }
  // [I]: all shards (4 shards for llama2, 5 shards for llama)
  {
    // [I]: input_ids
    Result<TensorInfo> input_ids = methods_meta[0]->input_tensor_meta(0);
    input_ids_ = std::make_unique<TensorImpl>(
        input_ids->scalar_type(),
        input_ids->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_ids->sizes().data()),
        &ptr->input_ids,
        const_cast<TensorImpl::DimOrderType*>(input_ids->dim_order().data()));
    input_tensors_[0].push_back(input_ids_.get());
    // [I]: atten_mask
    Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(1);
    attention_mask_ = std::make_unique<TensorImpl>(
        atten_mask->scalar_type(),
        atten_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
        ptr->attention_mask,
        const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
    input_tensors_[0].push_back(attention_mask_.get());
    // [I]: pos_ids_cos
    Result<TensorInfo> pos_ids_cos = methods_meta[0]->input_tensor_meta(2);
    position_ids_cos_ = std::make_unique<TensorImpl>(
        pos_ids_cos->scalar_type(),
        pos_ids_cos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(pos_ids_cos->sizes().data()),
        ptr->position_ids_cos,
        const_cast<TensorImpl::DimOrderType*>(pos_ids_cos->dim_order().data()));
    input_tensors_[0].push_back(position_ids_cos_.get());
    // [I]: pos_ids_sin
    Result<TensorInfo> pos_ids_sin = methods_meta[0]->input_tensor_meta(3);
    position_ids_sin_ = std::make_unique<TensorImpl>(
        pos_ids_sin->scalar_type(),
        pos_ids_sin->sizes().size(),
        const_cast<TensorImpl::SizesType*>(pos_ids_sin->sizes().data()),
        ptr->position_ids_sin,
        const_cast<TensorImpl::DimOrderType*>(pos_ids_sin->dim_order().data()));
    input_tensors_[0].push_back(position_ids_sin_.get());
    // [IO]: hidden_state => [I] shard2,3,4
    int output_index =
        shard_layers_[0] * 2 * num_heads_; // layers*(k + v caches)*heads
    Result<TensorInfo> hidden_state =
        methods_meta[0]->output_tensor_meta(output_index);
    hidden_state_ = std::make_unique<TensorImpl>(
        hidden_state->scalar_type(),
        hidden_state->sizes().size(),
        const_cast<TensorImpl::SizesType*>(hidden_state->sizes().data()),
        ptr->hidden_state,
        const_cast<TensorImpl::DimOrderType*>(
            hidden_state->dim_order().data()));
    // reuse inputs for following tensors
    for (int shard_index = 1; shard_index < modules_.size(); ++shard_index) {
      // inputs of shards 1 to n: hidden_state, atten_mask, pos_ids_cos,
      // pos_ids_sin
      input_tensors_[shard_index].push_back(hidden_state_.get());
      input_tensors_[shard_index].push_back(attention_mask_.get());
      input_tensors_[shard_index].push_back(position_ids_cos_.get());
      input_tensors_[shard_index].push_back(position_ids_sin_.get());
    }
  }
  // [I] kv_cache for all shards (4 shards for llama2 and 5 shards for llama3)
  for (int offset = 0, shard_index = 0, v_stride = 1023 * 128;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
      for (int cache_group = 0; cache_group < 2; ++cache_group) {
        for (int head = 0; head < num_heads_; ++head) {
          // bypass hidden_state(input_ids), atten_mask, pos_cos, pos_sin
          int index =
              num_heads_ * 2 * layer + cache_group * num_heads_ + head + 4;
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
  // [O] kv_cache for all shards (4 shards for llama2 and 5 shards for llama3)
  for (int offset = 0, shard_index = 0, v_stride = 1023 * 128;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
      for (int cache_group = 0; cache_group < 2; ++cache_group) {
        for (int head = 0; head < num_heads_; ++head) {
          int index = num_heads_ * 2 * layer + cache_group * num_heads_ + head;
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? v_cache_out_ : k_cache_out_);

          void* cache_ptr = (cache_group == 0)
              ? static_cast<void*>(
                    ptr->v_cache[layer + offset] + (head + 1) * v_stride)
              : static_cast<void*>(ptr->k_cache_out[layer + offset][head]);

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
  // [O]: hidden_state for shard 0 to n-1
  for (int shard_index = 0; shard_index < modules_.size() - 1; ++shard_index) {
    output_tensors_[shard_index].push_back(hidden_state_.get());
  }
  // [O]: logits
  {
    int output_index = shard_layers_[modules_.size() - 1] * 2 *
        num_heads_; // layers*(k + v caches)*heads
    Result<TensorInfo> logits =
        methods_meta[modules_.size() - 1]->output_tensor_meta(output_index);
    logits_ = std::make_unique<TensorImpl>(
        logits->scalar_type(),
        logits->sizes().size(),
        const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
        ptr->logits,
        const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
    output_tensors_[modules_.size() - 1].push_back(logits_.get());
  }

  // QAIHub Llama2 have 4* io compared to QAIHub Llama3,
  // so we use multi-threading for Llama2 when updating io
  if (num_heads_ == 32) {
    // thread pool jobs
    for (int i = 0, range = 1024 / thread_pool_.num_workers();
         i < thread_pool_.num_workers();
         ++i) {
      lr_update_kv_.push_back(
          {.start = i * range, .end = (i + 1) * range, .step = 1});
    }
  }
}

void KVCachedMemory::update_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  int seq_len = 1023;
  // update input_ids
  ptr->input_ids = static_cast<int32_t>(cur_token);
  // update causal mask for next token
  ptr->attention_mask[seq_len - pos] = 65535;
  // update position_ids
  position_ids_cos_->set_data(position_ids_cos_->mutable_data<uint16_t>() + 64);
  position_ids_sin_->set_data(position_ids_sin_->mutable_data<uint16_t>() + 64);

  // use multithreading when we have a lot of ios, Llama2 in this case
  if (num_heads_ == 32) {
    auto update_kv = [&](void* arg) {
      LoopRange* lr = static_cast<LoopRange*>(arg);
      // update v_cache
      for (int i = lr->start; i < lr->end; i += lr->step) {
        v_cache_in_[i]->set_data(v_cache_in_[i]->mutable_data<uint8_t>() + 128);
        v_cache_out_[i]->set_data(
            v_cache_out_[i]->mutable_data<uint8_t>() + 128);
      }
      // update output tensors of v_cache, 256 is the number of kvs per shard
      int shard = lr->start >> 8, offset = shard << 8;
      int start = lr->start - offset, end = lr->end - offset;
      for (int cache_stride = start; cache_stride < end; cache_stride += 32) {
        for (int cache_group = 0; cache_group < 2; ++cache_group) {
          for (int head = 0; head < 32; ++head) {
            // k, v are placed interleaved
            int index = (cache_stride << 1) + (cache_group << 5) + head;
            ET_CHECK_MSG(
                modules_[shard]->set_output(
                    output_tensors[shard][index], index) == Error::Ok,
                "failed to set output tensor for module %d's %d'th output "
                "while updating kv_cache output tensors",
                shard,
                index);
          }
        }
      }
    };

    for (int i = 0; i < lr_update_kv_.size(); ++i) {
      futures_[i] = std::move(thread_pool_.issue(update_kv, &lr_update_kv_[i]));
    }
  } else {
    // update v_cache
    for (int i = 0; i < v_cache_in_.size(); i++) {
      v_cache_in_[i]->set_data(v_cache_in_[i]->mutable_data<uint8_t>() + 128);
      v_cache_out_[i]->set_data(v_cache_out_[i]->mutable_data<uint8_t>() + 128);
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
  }
  // update k_cache by single thread, this part is cpu cache sensitive
  for (int i = 0; i < k_cache_in_.size(); ++i) {
    uint8_t* ptr_in = k_cache_in_[i]->mutable_data<uint8_t>();
    const uint8_t* ptr_out = k_cache_out_[i]->data<uint8_t>();
    for (size_t j = 0, offset = seq_len; j < 128; ++j, offset += seq_len) {
      ptr_in[offset] = ptr_out[j];
    }
    k_cache_in_[i]->set_data(ptr_in + 1);
  }
  for (auto& future : futures_) {
    future.wait();
  }
}

ThreadPool::ThreadPool() : stop_(false) {
  size_t hc = (std::thread::hardware_concurrency() + 3) / 4;
  // maximum number should be divisible by head dimension which equals to 32
  num_workers_ = std::min<size_t>(32, hc * 4);
  for (size_t i = 0; i < num_workers_; ++i) {
    threads_.emplace_back([this]() {
      while (1) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !jobs_.empty() || stop_; });

        if (stop_ && jobs_.empty())
          return;

        JobInfo job_info(std::move(jobs_.front()));
        jobs_.pop();
        lock.unlock();
        job_info.func(job_info.arg);
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  std::unique_lock<std::mutex> lock(mutex_);
  stop_ = true;
  lock.unlock();
  cv_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

std::future<void> ThreadPool::issue(
    std::function<void(void*)> func,
    void* arg) {
  std::unique_lock<std::mutex> lock(mutex_);
  jobs_.push(JobInfo(std::packaged_task<void(void*)>(func), arg));
  std::future<void> f = std::move(jobs_.back().func.get_future());
  lock.unlock();
  cv_.notify_one();
  return f;
}

size_t ThreadPool::num_workers() {
  return num_workers_;
}

} // namespace example
