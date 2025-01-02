/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama3_2/runner/io_memory.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <algorithm>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

Memory::Memory(std::vector<std::shared_ptr<Module>>& modules)
    : data_ptr_(nullptr, [](void*) {}), modules_(modules) {}

Memory::~Memory() {}

void* Memory::get_mutable_ptr() {
  return data_ptr_.get();
}

std::vector<Tensor> Memory::get_input_tensors(
    int shard_index,
    const std::string& method_name) {
  std::vector<Tensor> ret;
  ret.reserve(input_tensors_.size());
  for (TensorImpl* impl : input_tensors_[method_name][shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

std::vector<Tensor> Memory::get_output_tensors(
    int shard_index,
    const std::string& method_name) {
  std::vector<Tensor> ret;
  ret.reserve(output_tensors_[method_name][shard_index].size());
  for (TensorImpl* impl : output_tensors_[method_name][shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

HybridMemory::HybridMemory(
    std::vector<std::shared_ptr<Module>>& modules,
    int32_t prefill_cache_len,
    int32_t kv_cache_len,
    int32_t vocab_size,
    int32_t num_layers,
    int32_t head_dim,
    int32_t num_heads,
    EvalMode eval_mode,
    const std::string& prefill_forward_name,
    const std::string& kv_forward_name)
    : Memory(modules),
      shard_layers_({num_layers}),
      kv_cache_len_(kv_cache_len),
      prefill_cache_len_(prefill_cache_len),
      vocab_size_(vocab_size),
      num_layers_(num_layers),
      head_dim_(head_dim),
      num_heads_(num_heads),
      eval_mode_(eval_mode),
      prefill_forward_name_(prefill_forward_name),
      kv_forward_name_(kv_forward_name) {
  if (!prefill_forward_name_.empty()) {
    input_tensors_[prefill_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
    output_tensors_[prefill_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
    k_cache_in_[prefill_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    v_cache_in_[prefill_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    k_cache_out_[prefill_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    v_cache_out_[prefill_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
  }
  if (!kv_forward_name_.empty()) {
    input_tensors_[kv_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
    output_tensors_[kv_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
    k_cache_in_[kv_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    v_cache_in_[kv_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    k_cache_out_[kv_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
    v_cache_out_[kv_forward_name_] =
        std::vector<std::unique_ptr<executorch::aten::TensorImpl>>();
  }

  data_ptr_ = std::unique_ptr<void, void (*)(void*)>(
      new IO, [](void* ptr) { delete static_cast<IO*>(ptr); });
}

void HybridMemory::init_io() {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::memset(ptr, 0, sizeof(IO));

  int32_t max_cache_len = std::max(kv_cache_len_, prefill_cache_len_);
  int32_t k_in_size = (head_dim_ + 1) * max_cache_len;
  int32_t v_cache_size = (num_heads_ + 1) * max_cache_len * head_dim_;
  int32_t k_cache_out_size = num_heads_ * head_dim_;
  if (eval_mode_ == EvalMode::kHybrid || eval_mode_ == EvalMode::kPrefill) {
    k_cache_out_size *= prefill_cache_len_;
  }

  // Init kv vector shape, general enough to be shared across all 3 modes.
  ptr->k_cache_out.reserve(num_layers_);
  ptr->v_cache.reserve(num_layers_);
  for (int layer = 0; layer < num_layers_; layer++) {
    ptr->k_cache_out.emplace_back(std::vector<uint8_t>(k_cache_out_size));
    ptr->v_cache.emplace_back(std::vector<uint8_t>(v_cache_size));
  }

  auto init_prefill = [&]() {
    ptr->prefill_input_toks.resize(prefill_cache_len_);
    ptr->prefill_atten_mask.resize(prefill_cache_len_ * prefill_cache_len_);
    ptr->prefill_logits.resize(prefill_cache_len_ * vocab_size_);
  };

  auto init_kv = [&]() {
    ptr->kv_logits.resize(vocab_size_);
    ptr->kv_attention_mask.resize((kv_cache_len_ + 1), -255);
    ptr->k_cache.reserve(num_layers_);
    for (int layer = 0; layer < num_layers_; layer++) {
      ptr->k_cache.emplace_back();
      ptr->k_cache[layer].reserve(num_heads_);
      for (int head = 0; head < num_heads_; head++) {
        ptr->k_cache[layer].emplace_back(std::vector<uint8_t>(k_in_size));
      }
    }
  };

  switch (eval_mode_) {
    case EvalMode::kPrefill:
      init_prefill();
      break;
    case EvalMode::kKVCached:
      init_kv();
      break;
    case EvalMode::kHybrid:
      init_prefill();
      init_kv();
      break;
    default:
      break;
  }
}

void HybridMemory::prepare_kv_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  ET_CHECK_MSG(!(kv_forward_name_.empty()), "kv forward name is empty");
  IO* ptr = static_cast<IO*>(data_ptr_.get());

  // [I]: input_tokens
  Result<TensorInfo> input_tok = methods_meta[0]->input_tensor_meta(0);
  input_tok_ = std::make_unique<TensorImpl>(
      input_tok->scalar_type(),
      input_tok->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_tok->sizes().data()),
      &ptr->input_tok,
      const_cast<TensorImpl::DimOrderType*>(input_tok->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(input_tok_.get());

  // [I]: atten_mask
  Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(1);
  attention_mask_ = std::make_unique<TensorImpl>(
      atten_mask->scalar_type(),
      atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
      ptr->kv_attention_mask.data(),
      const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(attention_mask_.get());

  // [I]: input_pos
  Result<TensorInfo> input_pos = methods_meta[0]->input_tensor_meta(2);
  input_pos_ = std::make_unique<TensorImpl>(
      input_pos->scalar_type(),
      input_pos->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
      &ptr->input_pos,
      const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(input_pos_.get());

  // [I] kv_cache
  int index = 3; // bypass input_tokens, input_pos, atten_mask
  for (int offset = 0, shard_index = 0, v_stride = kv_cache_len_ * head_dim_;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->input_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_in_[kv_forward_name_]
                                : v_cache_in_[kv_forward_name_]);
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
          input_tensors_[kv_forward_name_][shard_index].push_back(
              cache.back().get());
        }
      }
    }
  }

  // [O]: logits
  int logit_index = 0;
  Result<TensorInfo> logits =
      methods_meta[modules_.size() - 1]->output_tensor_meta(logit_index);
  kv_logits_ = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      ptr->kv_logits.data(),
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[kv_forward_name_][modules_.size() - 1].push_back(
      kv_logits_.get());

  // [O] kv_cache
  index = 1;
  // Iterate through all kv cache outputs.
  // For k, we store it in k_cache_out and update to k_cache later.
  // For v, we append the output to the end of v_cache,
  // which serves as both input and output.
  for (int offset = 0, shard_index = 0, v_stride = kv_cache_len_ * head_dim_;
       shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_out_[kv_forward_name_]
                                : v_cache_out_[kv_forward_name_]);
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
          output_tensors_[kv_forward_name_][shard_index].push_back(
              cache.back().get());
        }
      }
    }
  }
}

void HybridMemory::prepare_prefill_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  ET_CHECK_MSG(
      !(prefill_forward_name_.empty()), "prefill forward name is empty");

  IO* ptr = static_cast<IO*>(data_ptr_.get());

  // [I]: pre_input_tokens
  Result<TensorInfo> prefill_input_toks = methods_meta[0]->input_tensor_meta(0);
  prefill_input_toks_ = std::make_unique<TensorImpl>(
      prefill_input_toks->scalar_type(),
      prefill_input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_input_toks->sizes().data()),
      ptr->prefill_input_toks.data(),
      const_cast<TensorImpl::DimOrderType*>(
          prefill_input_toks->dim_order().data()));
  input_tensors_[prefill_forward_name_][0].push_back(prefill_input_toks_.get());
  // [I]: prefill_attn_mask
  for (int i = 0; i < prefill_cache_len_; ++i) {
    for (int j = 0; j < prefill_cache_len_; ++j) {
      if (i < j) {
        ptr->prefill_atten_mask[i * prefill_cache_len_ + j] = -255;
      } else {
        ptr->prefill_atten_mask[i * prefill_cache_len_ + j] = 0;
      }
    }
  }
  Result<TensorInfo> prefill_atten_mask = methods_meta[0]->input_tensor_meta(1);
  prefill_attn_mask_ = std::make_unique<TensorImpl>(
      prefill_atten_mask->scalar_type(),
      prefill_atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_atten_mask->sizes().data()),
      ptr->prefill_atten_mask.data(),
      const_cast<TensorImpl::DimOrderType*>(
          prefill_atten_mask->dim_order().data()));
  input_tensors_[prefill_forward_name_][0].push_back(prefill_attn_mask_.get());
  // [O]: logits
  int logit_index = 0;
  Result<TensorInfo> logits =
      methods_meta[modules_.size() - 1]->output_tensor_meta(logit_index);
  prefill_logits_ = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      ptr->prefill_logits.data(),
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[prefill_forward_name_][modules_.size() - 1].push_back(
      prefill_logits_.get());

  // [O] kv_cache
  int index = 1;
  // prefill_k_stride should be equal to prefill_v_stride in prefill mode.
  // In hybrid mode, we use kv mode cache len for v stride since we want to
  // update prefill's result onto kv modes input.
  int32_t prefill_k_stride = prefill_cache_len_ * head_dim_;
  int32_t prefill_v_stride =
      std::max(prefill_cache_len_, kv_cache_len_) * head_dim_;

  if (eval_mode_ == EvalMode::kPrefill) {
    ET_CHECK_MSG(
        prefill_k_stride == prefill_v_stride,
        "prefill_k_stride should be equal to prefill_v_stride");
  }
  for (int offset = 0, shard_index = 0; shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_out_[prefill_forward_name_]
                                : v_cache_out_[prefill_forward_name_]);
          void* cache_ptr = (cache_group == 0)
              ? static_cast<void*>(
                    ptr->k_cache_out[layer + offset].data() +
                    head * prefill_k_stride)
              : static_cast<void*>(
                    ptr->v_cache[layer + offset].data() +
                    (head + 1) * prefill_v_stride);
          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          output_tensors_[prefill_forward_name_][shard_index].push_back(
              cache.back().get());
        }
      }
    }
  }
}

void HybridMemory::update_prefill_to_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  ET_CHECK_MSG(kv_cache_len_ != 0, "k_cache_len_ should not equal to 0");
  ET_CHECK_MSG(
      prefill_cache_len_ != 0, "prefill_cache_len_ should not equal to 0");
  IO* ptr = static_cast<IO*>(data_ptr_.get());

  ptr->input_tok = static_cast<int32_t>(cur_token);
  ptr->input_pos = static_cast<int32_t>(pos);
  // If prompt len is 30, prefill will handle to pos = 30.
  // At this point, pos should be 31.
  for (int i = 0; i < pos + 1; i++) {
    ptr->kv_attention_mask[kv_cache_len_ - i] = 0;
  }

  // update v_cache
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>>& v_cache_in =
      v_cache_in_[kv_forward_name_];
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>>& v_cache_out =
      v_cache_out_[kv_forward_name_];
  for (int i = 0, v_cache_stride = head_dim_ * pos; i < v_cache_in.size();
       i++) {
    v_cache_in[i]->set_data(
        v_cache_in[i]->mutable_data<uint8_t>() + v_cache_stride);
    v_cache_out[i]->set_data(
        v_cache_out[i]->mutable_data<uint8_t>() + v_cache_stride);
  }
  for (int shard = 0; shard < output_tensors.size(); shard++) {
    for (int index = 0; index < output_tensors[shard].size(); index++) {
      ET_CHECK_MSG(
          modules_[shard]->set_output(
              kv_forward_name_, output_tensors[shard][index], index) ==
              Error::Ok,
          "Failed to set output tensor for module %d's %d'th output "
          "while updating kv_cache output tensors",
          shard,
          index);
    }
  }

  std::vector<std::unique_ptr<executorch::aten::TensorImpl>>& k_cache_in =
      k_cache_in_[kv_forward_name_];
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>>& k_cache_out =
      k_cache_out_[prefill_forward_name_];
  for (int i = 0; i < k_cache_in.size(); ++i) {
    uint8_t* ptr_in = k_cache_in[i]->mutable_data<uint8_t>();
    const uint8_t* ptr_out = k_cache_out[i]->data<uint8_t>();
    for (size_t j = 0, offset = kv_cache_len_; j < head_dim_;
         ++j, offset += kv_cache_len_) {
      for (int k = 0, k_stride = j * prefill_cache_len_; k < pos; k++) {
        ptr_in[offset + k] = ptr_out[k_stride + k];
      }
    }
    k_cache_in[i]->set_data(ptr_in + pos);
  }
}

void HybridMemory::update_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  // update input_tok
  ptr->input_tok = static_cast<int32_t>(cur_token);
  // update position_ids
  ptr->input_pos = static_cast<int32_t>(pos);
  // update causal mask for next token
  ptr->kv_attention_mask[kv_cache_len_ - pos] = 0;

  // update v_cache
  auto& v_cache_in = v_cache_in_[kv_forward_name_];
  auto& v_cache_out = v_cache_out_[kv_forward_name_];
  for (int i = 0; i < v_cache_in.size(); i++) {
    v_cache_in[i]->set_data(v_cache_in[i]->mutable_data<uint8_t>() + head_dim_);
    v_cache_out[i]->set_data(
        v_cache_out[i]->mutable_data<uint8_t>() + head_dim_);
  }

  for (int shard = 0; shard < output_tensors.size(); shard++) {
    for (int index = 0; index < output_tensors[shard].size(); index++) {
      ET_CHECK_MSG(
          modules_[shard]->set_output(
              kv_forward_name_, output_tensors[shard][index], index) ==
              Error::Ok,
          "failed to set output tensor for module %d's %d'th output "
          "while updating kv_cache output tensors",
          shard,
          index);
    }
  }

  auto& k_cache_in = k_cache_in_[kv_forward_name_];
  auto& k_cache_out = k_cache_out_[kv_forward_name_];
  // update k_cache by single thread, this part is cpu cache sensitive
  for (int i = 0; i < k_cache_in.size(); ++i) {
    uint8_t* ptr_in = k_cache_in[i]->mutable_data<uint8_t>();
    const uint8_t* ptr_out = k_cache_out[i]->data<uint8_t>();
    for (size_t j = 0, offset = kv_cache_len_; j < head_dim_;
         ++j, offset += kv_cache_len_) {
      ptr_in[offset] = ptr_out[j];
    }
    k_cache_in[i]->set_data(ptr_in + 1);
  }
}

} // namespace example
