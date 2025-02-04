/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/io_manager.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <algorithm>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

IoMgrBase::IoMgrBase(std::vector<std::shared_ptr<Module>>& modules)
    : data_ptr_(nullptr, [](void*) {}), modules_(modules) {}

IoMgrBase::~IoMgrBase() {}

void* IoMgrBase::get_mutable_ptr() {
  return data_ptr_.get();
}

std::vector<Tensor> IoMgrBase::get_input_tensors(
    int shard_index,
    const std::string& method_name) {
  std::vector<Tensor> ret;
  ret.reserve(input_tensors_.size());
  for (TensorImpl* impl : input_tensors_[method_name][shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

std::vector<Tensor> IoMgrBase::get_output_tensors(
    int shard_index,
    const std::string& method_name) {
  std::vector<Tensor> ret;
  ret.reserve(output_tensors_[method_name][shard_index].size());
  for (TensorImpl* impl : output_tensors_[method_name][shard_index]) {
    ret.emplace_back(Tensor(impl));
  }
  return ret;
}

ShiftPointerIoMgr::ShiftPointerIoMgr(
    std::vector<std::shared_ptr<Module>>& modules,
    int32_t prefill_cache_len,
    int32_t kv_cache_len,
    int32_t vocab_size,
    int32_t num_layers,
    int32_t head_dim,
    int32_t num_heads,
    EvalMode eval_mode,
    const std::string& prefill_forward_name,
    const std::string& kv_forward_name,
    const bool use_int64_token)
    : IoMgrBase(modules),
      shard_layers_({num_layers}),
      kv_cache_len_(kv_cache_len),
      prefill_cache_len_(prefill_cache_len),
      vocab_size_(vocab_size),
      num_layers_(num_layers),
      head_dim_(head_dim),
      num_heads_(num_heads),
      eval_mode_(eval_mode),
      prefill_forward_name_(prefill_forward_name),
      kv_forward_name_(kv_forward_name),
      use_int64_token_(use_int64_token) {
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

void ShiftPointerIoMgr::init_io() {
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
    ptr->kv_attention_mask.resize((kv_cache_len_ + 1), 0);
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

void ShiftPointerIoMgr::prepare_kv_io(
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

void ShiftPointerIoMgr::prepare_prefill_io(
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
        ptr->prefill_atten_mask[i * prefill_cache_len_ + j] = 0;
      } else {
        ptr->prefill_atten_mask[i * prefill_cache_len_ + j] = 65535;
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

void ShiftPointerIoMgr::update_prefill_to_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  ET_CHECK_MSG(kv_cache_len_ != 0, "k_cache_len_ should not equal to 0");
  ET_CHECK_MSG(
      prefill_cache_len_ != 0, "prefill_cache_len_ should not equal to 0");
  IO* ptr = static_cast<IO*>(data_ptr_.get());

  ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  ptr->input_pos = static_cast<int32_t>(pos);
  // If prompt len is 30, prefill will handle to pos = 30.
  // At this point, pos should be 31.
  for (int i = 0; i < pos + 1; i++) {
    ptr->kv_attention_mask[kv_cache_len_ - i] = 65535;
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

void ShiftPointerIoMgr::update_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  // update input_tok
  ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  // update position_ids
  ptr->input_pos = static_cast<int32_t>(pos);
  // update causal mask for next token
  ptr->kv_attention_mask[kv_cache_len_ - pos] = 65535;

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

void ShiftPointerIoMgr::update_prefill_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  (void)output_tensors;
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  // Support CPU 4-bit embedding, which requires int64 input.
  // However, for QNN embedding, only int32 input is needed.
  // Therefore, we need to cast to the correct type to write the data.
  if (use_int64_token_) {
    ptr->prefill_input_toks[pos] = cur_token;
  } else {
    int32_t* prefill_input_toks_ptr =
        reinterpret_cast<int32_t*>(ptr->prefill_input_toks.data());
    prefill_input_toks_ptr[pos] = static_cast<int32_t>(cur_token);
  }
}

void ShiftPointerIoMgr::fill_prefill_toks(
    std::vector<uint64_t>& prompt_tokens) {
  IO* ptr = static_cast<IO*>(get_mutable_ptr());
  for (int i = 0; i < prompt_tokens.size(); i++) {
    // Support CPU 4-bit embedding, which requires int64 input.
    // However, for QNN embedding, only int32 input is needed.
    // Therefore, we need to cast to the correct type to write the data.
    if (use_int64_token_) {
      ptr->prefill_input_toks[i] = prompt_tokens[i];
    } else {
      int32_t* prefill_input_toks_ptr =
          reinterpret_cast<int32_t*>(ptr->prefill_input_toks.data());
      prefill_input_toks_ptr[i] = static_cast<int32_t>(prompt_tokens[i]);
    }
  }
}

void ShiftPointerIoMgr::fill_kv_tok_mask(int64_t pos, int64_t cur_token) {
  IO* ptr = static_cast<IO*>(get_mutable_ptr());
  ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  ptr->kv_attention_mask[kv_cache_len_] = 65535;
}

SmartMaskIoMgr::SmartMaskIoMgr(
    std::vector<std::shared_ptr<Module>>& modules,
    int32_t prefill_cache_len,
    int32_t kv_cache_len,
    int32_t vocab_size,
    int32_t num_layers,
    int32_t head_dim,
    int32_t num_heads,
    EvalMode eval_mode,
    const std::string& prefill_forward_name,
    const std::string& kv_forward_name,
    const bool use_int64_token)
    : IoMgrBase(modules),
      shard_layers_({num_layers}),
      prefill_cache_len_(prefill_cache_len),
      kv_cache_len_(kv_cache_len),
      vocab_size_(vocab_size),
      num_layers_(num_layers),
      head_dim_(head_dim),
      num_heads_(num_heads),
      eval_mode_(eval_mode),
      prefill_forward_name_(prefill_forward_name),
      kv_forward_name_(kv_forward_name),
      use_int64_token_(use_int64_token) {
  if (!prefill_forward_name_.empty()) {
    input_tensors_[prefill_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
    output_tensors_[prefill_forward_name_] =
        std::vector<std::vector<executorch::aten::TensorImpl*>>(modules.size());
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

std::unordered_map<std::string, size_t> SmartMaskIoMgr::get_io_elements() {
  size_t cache_len = std::max(kv_cache_len_, prefill_cache_len_);
  size_t cache_in_ele = num_layers_ * num_heads_ * head_dim_ * cache_len;
  size_t cache_out_ele = num_layers_ * num_heads_ * head_dim_;
  return std::unordered_map<std::string, size_t>{
      {"input_tok_ele", 1},
      {"input_pos_ele", 1},
      {"cache_in_ele", cache_in_ele},
      {"cache_out_ele", cache_out_ele},
      // 1 for the input prompt
      {"atten_mask_ele", cache_len + 1},
      {"kv_logits_ele", vocab_size_},
      {"prefill_input_toks_ele", prefill_cache_len_},
      {"prefill_atten_mask_ele", prefill_cache_len_ * prefill_cache_len_},
      {"prefill_logits_ele", prefill_cache_len_ * vocab_size_}};
}

std::unordered_map<std::string, size_t> SmartMaskIoMgr::get_io_bytes() {
  std::unordered_map<std::string, size_t> element_map = get_io_elements();
  auto align = [](size_t byte) {
    size_t alignment = MemoryAllocator::kDefaultAlignment;
    return byte % alignment == 0 ? byte
                                 : byte +
            (static_cast<intptr_t>(alignment) -
             byte % static_cast<intptr_t>(alignment));
  };
  return std::unordered_map<std::string, size_t>{
      {"input_tok_bytes",
       align(element_map["input_tok_ele"] * sizeof(int32_t))},
      {"input_pos_bytes",
       align(element_map["input_pos_ele"] * sizeof(int32_t))},
      {"cache_in_bytes", align(element_map["cache_in_ele"] * sizeof(uint8_t))},
      {"cache_out_bytes",
       align(element_map["cache_out_ele"] * sizeof(uint8_t))},
      {"atten_mask_bytes",
       align(element_map["atten_mask_ele"] * sizeof(uint16_t))},
      {"kv_logits_bytes",
       align(element_map["kv_logits_ele"] * sizeof(uint16_t))},
      {"prefill_input_toks_bytes",
       align(element_map["prefill_input_toks_ele"] * sizeof(int32_t))},
      {"prefill_atten_mask_bytes",
       align(element_map["prefill_atten_mask_ele"] * sizeof(uint16_t))},
      {"prefill_logits_bytes",
       align(element_map["prefill_logits_ele"] * sizeof(uint16_t))}};
}

void SmartMaskIoMgr::IO::init_io_ptrs(
    void* shared_buffer_ptr,
    std::unordered_map<std::string, size_t>& io_bytes_map) {
  shared_buffer_base = shared_buffer_ptr;
  std::byte* cur_ptr = reinterpret_cast<std::byte*>(shared_buffer_base);
  std::size_t cur_pos = 0;
  size_t layered_head_count = num_layers_ * num_heads_;

  // Iterate map so that we don't need to care about which mode is used.
  for (const auto& iter : io_bytes_map) {
    std::string key = iter.first;
    size_t size = iter.second;
    if (key == "input_tok_bytes") {
      input_tok = reinterpret_cast<int64_t*>(cur_ptr);
    } else if (key == "input_pos_bytes") {
      input_pos = reinterpret_cast<int32_t*>(cur_ptr);
    } else if (key == "cache_in_bytes" || key == "cache_out_bytes") {
      auto& k_cache_ref = (key == "cache_in_bytes") ? k_cache : k_cache_out;
      auto& v_cache_ref = (key == "cache_in_bytes") ? v_cache : v_cache_out;
      size_t single_head_size = size / layered_head_count;
      k_cache_ref.reserve(num_layers_);
      v_cache_ref.reserve(num_layers_);
      for (int i = 0; i < num_layers_; ++i) {
        k_cache_ref[i].reserve(num_heads_);
        v_cache_ref[i].reserve(num_heads_);
        for (int j = 0; j < num_heads_; ++j) {
          k_cache_ref[i][j] = reinterpret_cast<uint8_t*>(cur_ptr);
          io_pos_map[cur_ptr] = cur_pos;
          cur_ptr += single_head_size;
          cur_pos += single_head_size;
          v_cache_ref[i][j] = reinterpret_cast<uint8_t*>(cur_ptr);
          io_pos_map[cur_ptr] = cur_pos;
          cur_ptr += single_head_size;
          cur_pos += single_head_size;
        }
      }
      continue;
    } else if (key == "atten_mask_bytes") {
      kv_attention_mask = reinterpret_cast<uint16_t*>(cur_ptr);
    } else if (key == "kv_logits_bytes") {
      kv_logits = reinterpret_cast<uint16_t*>(cur_ptr);
    } else if (key == "prefill_input_toks_bytes") {
      prefill_input_toks = reinterpret_cast<int64_t*>(cur_ptr);
    } else if (key == "prefill_atten_mask_bytes") {
      prefill_atten_mask = reinterpret_cast<uint16_t*>(cur_ptr);
    } else if (key == "prefill_logits_bytes") {
      prefill_logits = reinterpret_cast<uint16_t*>(cur_ptr);
    } else {
      ET_LOG(Error, "Unknown pointer type: %s", key.c_str());
    }

    io_pos_map[cur_ptr] = cur_pos;
    cur_ptr += size;
    cur_pos += size;
  }
}

void SmartMaskIoMgr::IO::add_custom_mem_info(
    void* ptr,
    size_t nbytes,
    executorch::aten::ScalarType scalar_type,
    executorch::runtime::TensorInfo& tensor_info) {
  if (auto it = io_pos_map.find(static_cast<std::byte*>(ptr));
      it == io_pos_map.end()) {
    ET_LOG(Error, "Shared buffer pointer %p is not found", ptr);
  }
  size_t pos = io_pos_map[static_cast<std::byte*>(ptr)];
  uint32_t rank = tensor_info.sizes().size();
  uint32_t shape[rank];
  CustomMemTensorInfo info = {
      shared_buffer_base, ptr, pos, nbytes, shape, rank, scalar_type};
  QnnExecuTorchAddCustomMemTensorInfo(info);
}

void SmartMaskIoMgr::init_io() {
  std::unordered_map<std::string, size_t> io_bytes_map = get_io_bytes();

  switch (eval_mode_) {
    case EvalMode::kPrefill:
      io_bytes_map.erase("input_tok_bytes");
      io_bytes_map.erase("input_pos_bytes");
      io_bytes_map.erase("atten_mask_bytes");
      io_bytes_map.erase("kv_logits_bytes");
      break;
    case EvalMode::kKVCached:
      io_bytes_map.erase("prefill_input_toks_bytes");
      io_bytes_map.erase("prefill_atten_mask_bytes");
      io_bytes_map.erase("prefill_logits_bytes");
      break;
    case EvalMode::kHybrid:
      break;
    default:
      break;
  }

  size_t total_bytes = 0;
  for (const auto& iter : io_bytes_map) {
    size_t size = iter.second;
    if (iter.first == "cache_in_bytes" || iter.first == "cache_out_bytes") {
      size = iter.second * 2;
    }
    total_bytes += size;
  }
  void* shared_ptr = QnnExecuTorchAllocCustomMem(
      total_bytes, MemoryAllocator::kDefaultAlignment);

  ET_CHECK_MSG(
      shared_ptr,
      "Allocate Rpc mem falied, bytes=%zu, alignment=%zu",
      total_bytes,
      MemoryAllocator::kDefaultAlignment);
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  ptr->num_heads_ = num_heads_;
  ptr->num_layers_ = num_layers_;
  ptr->head_dim_ = head_dim_;
  ptr->init_io_ptrs(shared_ptr, io_bytes_map);
}

void SmartMaskIoMgr::prepare_kv_io(
    const std::vector<Result<MethodMeta>>& methods_meta) {
  for (int i = 0; i < modules_.size(); ++i) {
    ET_CHECK_MSG(
        methods_meta[i].ok(),
        "Failed to get method_meta 0x%x",
        static_cast<uint32_t>(methods_meta[i].error()));
  }

  ET_CHECK_MSG(!(kv_forward_name_.empty()), "kv forward name is empty");
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  std::unordered_map<std::string, size_t> io_bytes_map = get_io_bytes();

  // [I]: input_tokens
  Result<TensorInfo> input_tok = methods_meta[0]->input_tensor_meta(0);
  input_tok_ = std::make_unique<TensorImpl>(
      input_tok->scalar_type(),
      input_tok->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_tok->sizes().data()),
      ptr->input_tok,
      const_cast<TensorImpl::DimOrderType*>(input_tok->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(input_tok_.get());
  ptr->add_custom_mem_info(
      ptr->input_tok,
      io_bytes_map["input_tok_bytes"],
      input_tok->scalar_type(),
      input_tok.get());

  // [I]: atten_mask
  Result<TensorInfo> atten_mask = methods_meta[0]->input_tensor_meta(1);
  attention_mask_ = std::make_unique<TensorImpl>(
      atten_mask->scalar_type(),
      atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(atten_mask->sizes().data()),
      ptr->kv_attention_mask,
      const_cast<TensorImpl::DimOrderType*>(atten_mask->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(attention_mask_.get());
  ptr->add_custom_mem_info(
      ptr->kv_attention_mask,
      io_bytes_map["atten_mask_bytes"],
      atten_mask->scalar_type(),
      atten_mask.get());

  // [I]: input_pos
  Result<TensorInfo> input_pos = methods_meta[0]->input_tensor_meta(2);
  input_pos_ = std::make_unique<TensorImpl>(
      input_pos->scalar_type(),
      input_pos->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
      ptr->input_pos,
      const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
  input_tensors_[kv_forward_name_][0].push_back(input_pos_.get());
  ptr->add_custom_mem_info(
      ptr->input_pos,
      io_bytes_map["input_pos_bytes"],
      input_pos->scalar_type(),
      input_pos.get());

  // [I] kv_cache
  size_t layered_head_count = num_layers_ * num_heads_;
  int index = 3; // bypass input_tokens, input_pos, atten_mask
  for (int offset = 0, shard_index = 0; shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->input_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_in_[kv_forward_name_]
                                : v_cache_in_[kv_forward_name_]);
          uint8_t* cache_ptr = (cache_group == 0)
              ? ptr->k_cache[layer + offset][head]
              : ptr->v_cache[layer + offset][head];

          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          ptr->add_custom_mem_info(
              cache_ptr,
              io_bytes_map["cache_in_bytes"] / layered_head_count,
              kv_cache->scalar_type(),
              kv_cache.get());
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
      ptr->kv_logits,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));

  ptr->add_custom_mem_info(
      ptr->kv_logits,
      io_bytes_map["kv_logits_bytes"],
      logits->scalar_type(),
      logits.get());
  output_tensors_[kv_forward_name_][modules_.size() - 1].push_back(
      kv_logits_.get());

  // [O] kv_cache
  index = 1;
  for (int offset = 0, shard_index = 0; shard_index < modules_.size();
       offset += shard_layers_[shard_index], shard_index++) {
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      for (int layer = 0; layer < shard_layers_[shard_index]; ++layer) {
        for (int head = 0; head < num_heads_; ++head, ++index) {
          Result<TensorInfo> kv_cache =
              methods_meta[shard_index]->output_tensor_meta(index);
          std::vector<std::unique_ptr<TensorImpl>>& cache =
              (cache_group == 0 ? k_cache_out_[kv_forward_name_]
                                : v_cache_out_[kv_forward_name_]);
          uint8_t* cache_ptr = (cache_group == 0)
              ? ptr->k_cache_out[layer + offset][head]
              : ptr->v_cache_out[layer + offset][head];
          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          ptr->add_custom_mem_info(
              cache_ptr,
              io_bytes_map["cache_out_bytes"] / layered_head_count,
              kv_cache->scalar_type(),
              kv_cache.get());
          output_tensors_[kv_forward_name_][shard_index].push_back(
              cache.back().get());
        }
      }
    }
  }
}

void SmartMaskIoMgr::update_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  size_t cache_len = std::max(kv_cache_len_, prefill_cache_len_);
  // update input_tok
  *ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  // update position_ids
  *ptr->input_pos = static_cast<int32_t>(pos);
  // update smart mask for previous cache
  ptr->kv_attention_mask[pos] = 65535;

  // update v_cache
  auto& v_cache_in = v_cache_in_[kv_forward_name_];
  auto& v_cache_out = v_cache_out_[kv_forward_name_];
  // update v_cache by single thread, this part is cpu cache sensitive
  for (int i = 0; i < v_cache_in.size(); ++i) {
    uint8_t* ptr_in = v_cache_in[i]->mutable_data<uint8_t>() + pos * head_dim_;
    const uint8_t* ptr_out = v_cache_out[i]->data<uint8_t>();
    memcpy(ptr_in, ptr_out, head_dim_ * sizeof(uint8_t));
  }

  auto& k_cache_in = k_cache_in_[kv_forward_name_];
  auto& k_cache_out = k_cache_out_[kv_forward_name_];
  for (int i = 0; i < k_cache_in.size(); ++i) {
    uint8_t* ptr_in = k_cache_in[i]->mutable_data<uint8_t>() + pos;
    const uint8_t* ptr_out = k_cache_out[i]->data<uint8_t>();
    for (size_t j = 0, offset = 0; j < head_dim_; ++j, offset += cache_len) {
      ptr_in[offset] = ptr_out[j];
    }
  }
}

void SmartMaskIoMgr::prepare_prefill_io(
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
  std::unordered_map<std::string, size_t> io_bytes_map = get_io_bytes();

  int32_t cache_len = methods_meta[0]->input_tensor_meta(0)->sizes()[1];
  // [I]: pre_input_tokens
  Result<TensorInfo> prefill_input_toks = methods_meta[0]->input_tensor_meta(0);
  prefill_input_toks_ = std::make_unique<TensorImpl>(
      prefill_input_toks->scalar_type(),
      prefill_input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_input_toks->sizes().data()),
      ptr->prefill_input_toks,
      const_cast<TensorImpl::DimOrderType*>(
          prefill_input_toks->dim_order().data()));
  input_tensors_[prefill_forward_name_][0].push_back(prefill_input_toks_.get());
  ptr->add_custom_mem_info(
      ptr->prefill_input_toks,
      io_bytes_map["prefill_input_toks_bytes"],
      executorch::aten::ScalarType::Int,
      prefill_input_toks.get());

  // [I]: prefill_attn_mask
  for (int i = 0; i < cache_len; ++i) {
    for (int j = 0; j < cache_len; ++j) {
      if (i < j) {
        ptr->prefill_atten_mask[i * cache_len + j] = 0;
      } else {
        ptr->prefill_atten_mask[i * cache_len + j] = 65535;
      }
    }
  }
  Result<TensorInfo> prefill_atten_mask = methods_meta[0]->input_tensor_meta(1);
  prefill_attn_mask_ = std::make_unique<TensorImpl>(
      prefill_atten_mask->scalar_type(),
      prefill_atten_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(prefill_atten_mask->sizes().data()),
      ptr->prefill_atten_mask,
      const_cast<TensorImpl::DimOrderType*>(
          prefill_atten_mask->dim_order().data()));
  input_tensors_[prefill_forward_name_][0].push_back(prefill_attn_mask_.get());
  ptr->add_custom_mem_info(
      ptr->prefill_atten_mask,
      io_bytes_map["prefill_atten_mask_bytes"],
      executorch::aten::ScalarType::Bits16,
      prefill_atten_mask.get());

  // [O]: logits
  int logit_index = 0;
  Result<TensorInfo> logits = methods_meta[0]->output_tensor_meta(0);
  prefill_logits_ = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      ptr->prefill_logits,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_[prefill_forward_name_][modules_.size() - 1].push_back(
      prefill_logits_.get());
  ptr->add_custom_mem_info(
      ptr->prefill_logits,
      io_bytes_map["prefill_logits_bytes"],
      executorch::aten::ScalarType::Bits16,
      logits.get());

  // [O] kv_cache
  int index = 1;
  size_t layered_head_count = num_layers_ * num_heads_;
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
              ? ptr->k_cache[layer + offset][head]
              : ptr->v_cache[layer + offset][head];
          cache.emplace_back(std::make_unique<TensorImpl>(
              kv_cache->scalar_type(),
              kv_cache->sizes().size(),
              const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
              cache_ptr,
              const_cast<TensorImpl::DimOrderType*>(
                  kv_cache->dim_order().data())));
          ptr->add_custom_mem_info(
              cache_ptr,
              io_bytes_map["cache_in_bytes"] / layered_head_count,
              executorch::aten::ScalarType::Byte,
              kv_cache.get());
          output_tensors_[prefill_forward_name_][shard_index].push_back(
              cache.back().get());
        }
      }
    }
  }
}

void SmartMaskIoMgr::update_prefill_to_kv_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  IO* ptr = static_cast<IO*>(data_ptr_.get());

  *ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  *ptr->input_pos = static_cast<int32_t>(pos);
  // pos means the cur_token pos
  for (int i = 0; i < pos; i++) {
    ptr->kv_attention_mask[i] = 65535;
  }

  // Update K is enough, copy from last to prevent from overwriting values
  size_t copied_size = prefill_cache_len_ * sizeof(uint8_t);
  for (int l = 0; l < num_layers_; l++) {
    for (int h = 0; h < num_heads_; h++) {
      uint8_t* k_cache = ptr->k_cache[l][h];
      for (int hd = head_dim_ - 1; hd > -1; hd--) {
        memcpy(
            k_cache + (kv_cache_len_ * hd),
            k_cache + (prefill_cache_len_ * hd),
            copied_size);
      }
    }
  }
}

void SmartMaskIoMgr::update_prefill_io(
    int64_t cur_token,
    int64_t pos,
    std::vector<std::vector<Tensor>>& output_tensors) {
  (void)output_tensors;
  IO* ptr = static_cast<IO*>(data_ptr_.get());
  // Support CPU 4-bit embedding, which requires int64 input.
  // However, for QNN embedding, only int32 input is needed.
  // Therefore, we need to cast to the correct type to write the data.
  if (use_int64_token_) {
    ptr->prefill_input_toks[pos] = cur_token;
  } else {
    int32_t* prefill_input_toks_ptr =
        reinterpret_cast<int32_t*>(ptr->prefill_input_toks);
    prefill_input_toks_ptr[pos] = static_cast<int32_t>(cur_token);
  }
}

void SmartMaskIoMgr::fill_prefill_toks(std::vector<uint64_t>& prompt_tokens) {
  IO* ptr = static_cast<IO*>(get_mutable_ptr());
  for (int i = 0; i < prompt_tokens.size(); i++) {
    // Support CPU 4-bit embedding, which requires int64 input.
    // However, for QNN embedding, only int32 input is needed.
    // Therefore, we need to cast to the correct type to write the data.
    if (use_int64_token_) {
      ptr->prefill_input_toks[i] = prompt_tokens[i];
    } else {
      int32_t* prefill_input_toks_ptr =
          reinterpret_cast<int32_t*>(ptr->prefill_input_toks);
      prefill_input_toks_ptr[i] = static_cast<int32_t>(prompt_tokens[i]);
    }
  }
}

void SmartMaskIoMgr::fill_kv_tok_mask(int64_t pos, int64_t cur_token) {
  IO* ptr = static_cast<IO*>(get_mutable_ptr());
  *ptr->input_tok =
      use_int64_token_ ? cur_token : static_cast<int32_t>(cur_token);
  ptr->kv_attention_mask[kv_cache_len_] = 65535;
}

} // namespace example
