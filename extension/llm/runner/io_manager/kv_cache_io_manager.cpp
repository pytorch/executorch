/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/io_manager/kv_cache_io_manager.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <new>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch {
namespace extension {
namespace llm {

using ::executorch::aten::ScalarType;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;

namespace {

constexpr const char* kPromptMethod = "prefill_forward";
constexpr const char* kStepMethod = "kv_forward";
constexpr size_t kBufferAlignment = 64;

size_t align_up(size_t n) {
  return (n + kBufferAlignment - 1) / kBufferAlignment * kBufferAlignment;
}

size_t element_size(ScalarType dtype) {
  return ::executorch::runtime::elementSize(dtype);
}

template <typename T>
void fill(std::byte* base, size_t offset, size_t count, T value) {
  std::fill_n(reinterpret_cast<T*>(base) + offset, count, value);
}

void fill_value(
    ScalarType dtype,
    std::byte* base,
    size_t offset,
    size_t count,
    bool visible) {
  switch (dtype) {
    case ScalarType::Float:
      fill<float>(base, offset, count, visible ? 0.0f : -65535.0f);
      break;
    case ScalarType::UInt16:
      fill<uint16_t>(base, offset, count, visible ? 65535u : 0u);
      break;
    case ScalarType::Byte:
      fill<uint8_t>(base, offset, count, visible ? 255u : 0u);
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported attention mask dtype %d", (int)dtype);
  }
}

void write_index(ScalarType dtype, std::byte* base, size_t i, int64_t value) {
  if (dtype == ScalarType::Int) {
    reinterpret_cast<int32_t*>(base)[i] = static_cast<int32_t>(value);
  } else {
    reinterpret_cast<int64_t*>(base)[i] = value;
  }
}

} // namespace

KvCacheIOManager::KvCacheIOManager(ET_MODULE_NAMESPACE::Module& module)
    : IOManager(module), module_(module) {
  if (auto& factory = io_buffer_allocator_factory()) {
    allocator_ = factory();
  }
}

KvCacheIOManager::~KvCacheIOManager() {
  methods_.clear();
  past_k_.clear();
  past_v_.clear();
  for (const auto& block : blocks_) {
    if (block.from_allocator) {
      allocator_->deallocate(block.data);
    } else {
      ::operator delete(block.data, std::align_val_t(kBufferAlignment));
    }
  }
}

void KvCacheIOManager::allocate(
    const std::vector<std::pair<Buffer*, ::executorch::runtime::TensorInfo>>&
        buffers) {
  size_t total = 0;
  for (const auto& [buf, info] : buffers) {
    total = align_up(total) + info.nbytes();
  }
  void* block =
      allocator_ ? allocator_->allocate(total, kBufferAlignment) : nullptr;
  const bool from_allocator = block != nullptr;
  if (!from_allocator) {
    block = ::operator new(total, std::align_val_t(kBufferAlignment));
  }
  blocks_.push_back({block, from_allocator});
  std::memset(block, 0, total);
  size_t offset = 0;
  for (const auto& [buf, info] : buffers) {
    offset = align_up(offset);
    buf->data = static_cast<std::byte*>(block) + offset;
    buf->nbytes = info.nbytes();
    std::vector<executorch::aten::SizesType> sizes(
        info.sizes().begin(), info.sizes().end());
    buf->tensor = from_blob(buf->data, sizes, info.scalar_type());
    if (from_allocator) {
      allocator_->bind(block, buf->data);
    }
    offset += info.nbytes();
  }
}

std::pair<std::string, std::string> KvCacheIOManager::method_names(
    const std::unordered_set<std::string>& available) const {
  if (!available.count(kStepMethod)) {
    return {"", ""};
  }
  // A separate prompt graph takes the prompt in fixed-size chunks; without
  // one, the decode step serves the prompt too.
  return {
      available.count(kPromptMethod) ? kPromptMethod : kStepMethod,
      kStepMethod};
}

int64_t KvCacheIOManager::max_tokens_per_step(const std::string& method) const {
  auto it = methods_.find(method);
  return it == methods_.end() ? -1 : it->second.ar_len;
}

void KvCacheIOManager::rearrange(int32_t past_len) {
  // Move the filled positions into the layout of `past_len`: K rows and V
  // heads are strided by the past length, so each moves separately.
  if (active_past_len_ >= 0 && active_past_len_ != past_len && filled_ > 0) {
    const int32_t src_len = active_past_len_;
    for (int32_t l = 0; l < num_layers_; ++l) {
      auto& ks = past_k_[src_len][l];
      auto& kd = past_k_[past_len][l];
      const size_t es = element_size(ks.tensor->scalar_type());
      const size_t rows = ks.tensor->numel() / src_len;
      for (size_t r = 0; r < rows; ++r) {
        std::memcpy(
            kd.data + r * past_len * es,
            ks.data + r * src_len * es,
            filled_ * es);
      }
      auto& vs = past_v_[src_len][l];
      auto& vd = past_v_[past_len][l];
      const size_t heads = vs.tensor->size(1);
      const size_t dim = vs.tensor->size(3);
      for (size_t h = 0; h < heads; ++h) {
        std::memcpy(
            vd.data + h * past_len * dim * es,
            vs.data + h * src_len * dim * es,
            filled_ * dim * es);
      }
    }
  }
  active_past_len_ = past_len;
}

Error KvCacheIOManager::load(
    const std::string& prefill_method,
    const std::string& decode_method) {
  // Quantized logits are dequantized with the scale the program carries, if
  // any; without it they keep their order, which is all greedy decoding needs.
  auto names = module_.method_names();
  if (names.ok() && names->count("get_logits_scale") &&
      names->count("get_logits_zero_point")) {
    logits_scale_ = module_.get("get_logits_scale")->toDouble();
    logits_zero_point_ = module_.get("get_logits_zero_point")->toInt();
  }
  if (names.ok() && names->count("get_n_cross_layers")) {
    num_cross_layers_ =
        static_cast<int32_t>(module_.get("get_n_cross_layers")->toInt());
  }
  // The runner may also call load() with its default method names; only the
  // program's own methods are this contract's.
  for (const auto& method : {prefill_method, decode_method}) {
    if (names.ok() && names->count(method)) {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method));
    }
  }
  return Error::Ok;
}

Error KvCacheIOManager::load_method(const std::string& method) {
  if (methods_.count(method)) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module_.load_method(method));
  auto meta_result = module_.method_meta(method);
  ET_CHECK_OK_OR_RETURN_ERROR(meta_result.error());
  auto meta = std::move(*meta_result);
  const size_t num_inputs = meta.num_inputs();
  const size_t cross_inputs = 2 * static_cast<size_t>(num_cross_layers_);
  ET_CHECK_OR_RETURN_ERROR(
      num_inputs >= 5 + cross_inputs &&
          (num_inputs - 3 - cross_inputs) % 2 == 0,
      InvalidArgument,
      "%s has %zu inputs; a kvcache_io method takes tokens, mask, positions, a K and V cache per layer and %zu cross-attention inputs.",
      method.c_str(),
      num_inputs,
      cross_inputs);
  const int32_t layers =
      static_cast<int32_t>((num_inputs - 3 - cross_inputs) / 2);
  ET_CHECK_OR_RETURN_ERROR(
      num_layers_ == 0 || num_layers_ == layers,
      InvalidArgument,
      "%s has %d layers, another method %d.",
      method.c_str(),
      layers,
      num_layers_);
  num_layers_ = layers;

  MethodIO io;
  std::vector<std::pair<Buffer*, ::executorch::runtime::TensorInfo>> buffers;
  auto make = [&buffers](
                  Buffer& buf, const ::executorch::runtime::TensorInfo& info) {
    buffers.emplace_back(&buf, info);
  };
  auto tokens = *meta.input_tensor_meta(0);
  auto mask = *meta.input_tensor_meta(1);
  io.ar_len = tokens.sizes()[tokens.sizes().size() - 1];
  io.context_len = mask.sizes()[mask.sizes().size() - 1];
  io.past_len = io.context_len - io.ar_len;
  io.mask_dtype = mask.scalar_type();
  make(io.tokens, tokens);
  make(io.mask, mask);
  make(io.positions, *meta.input_tensor_meta(2));

  auto& past_k = past_k_[io.past_len];
  auto& past_v = past_v_[io.past_len];
  const bool fresh = past_k.empty();
  if (fresh) {
    past_k.resize(layers);
    past_v.resize(layers);
  }
  for (int32_t l = 0; l < layers; ++l) {
    if (fresh) {
      make(past_k[l], *meta.input_tensor_meta(3 + l));
      make(past_v[l], *meta.input_tensor_meta(3 + layers + l));
    }
    io.past_k.push_back(&past_k[l]);
    io.past_v.push_back(&past_v[l]);
  }

  make(io.logits, *meta.output_tensor_meta(0));
  io.new_k.resize(layers);
  io.new_v.resize(layers);
  for (int32_t l = 0; l < layers; ++l) {
    make(io.new_k[l], *meta.output_tensor_meta(1 + l));
    make(io.new_v[l], *meta.output_tensor_meta(1 + layers + l));
  }
  if (cross_.empty() && num_cross_layers_ > 0) {
    cross_.resize(cross_inputs);
    for (size_t i = 0; i < cross_inputs; ++i) {
      make(cross_[i], *meta.input_tensor_meta(3 + 2 * layers + i));
    }
  }
  allocate(buffers);
  // Outputs left out of memory planning are written into these buffers;
  // planned ones (e.g. a CPU program) come back in the program's own memory.
  auto bind = [&](const Buffer& buf, size_t index) -> Error {
    if (meta.output_tensor_meta(index)->is_memory_planned()) {
      return Error::Ok;
    }
    return module_.set_output(method, EValue(*buf.tensor), index);
  };
  ET_CHECK_OK_OR_RETURN_ERROR(bind(io.logits, 0));
  for (int32_t l = 0; l < layers; ++l) {
    ET_CHECK_OK_OR_RETURN_ERROR(bind(io.new_k[l], 1 + l));
    ET_CHECK_OK_OR_RETURN_ERROR(bind(io.new_v[l], 1 + layers + l));
  }
  const auto vocab = meta.output_tensor_meta(0)->sizes();
  logits_f_.assign(vocab[vocab.size() - 1], 0.0f);
  logits_f_tensor_ = from_blob(
      logits_f_.data(),
      {1, 1, static_cast<executorch::aten::SizesType>(logits_f_.size())},
      ScalarType::Float);
  methods_.emplace(method, std::move(io));
  return Error::Ok;
}

Error KvCacheIOManager::reset(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;
  for (auto* caches : {&past_k_, &past_v_}) {
    for (auto& [len, layers] : *caches) {
      for (auto& buf : layers) {
        std::memset(buf.data, 0, buf.nbytes);
      }
    }
  }
  filled_ = 0;
  active_past_len_ = -1;
  return Error::Ok;
}

void KvCacheIOManager::fill_mask(
    MethodIO& io,
    int64_t n_past,
    int32_t n_valid) {
  std::byte* base = io.mask.data;
  const size_t row = io.context_len;
  fill_value(io.mask_dtype, base, 0, row * io.ar_len, /*visible=*/false);
  for (int32_t r = 0; r < io.ar_len; ++r) {
    // Every token sees the filled past and the step's tokens up to itself.
    fill_value(io.mask_dtype, base, r * row, n_past, true);
    const int32_t self = std::min(r, std::max(n_valid - 1, 0));
    fill_value(io.mask_dtype, base, r * row + io.past_len, self + 1, true);
  }
}

Result<std::vector<EValue>> KvCacheIOManager::prepare(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& method) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method));
  auto& io = methods_.at(method);
  const int32_t n = static_cast<int32_t>(input->numel());
  const int64_t start = start_pos->const_data_ptr<int64_t>()[0];
  ET_CHECK_OR_RETURN_ERROR(
      n >= 1 && n <= io.ar_len,
      InvalidArgument,
      "%s takes at most %d tokens per step, got %d.",
      method.c_str(),
      io.ar_len,
      n);
  ET_CHECK_OR_RETURN_ERROR(
      start + n <= io.past_len + (io.ar_len == 1 ? 1 : 0),
      OutOfResources,
      "Context full: position %" PRId64 " with %d slots of past.",
      start,
      io.past_len);
  const auto token_dtype = io.tokens.tensor->scalar_type();
  const auto pos_dtype = io.positions.tensor->scalar_type();
  for (int32_t i = 0; i < io.ar_len; ++i) {
    const int32_t src = std::min(i, n - 1);
    const int64_t token = input->scalar_type() == ScalarType::Int
        ? input->const_data_ptr<int32_t>()[src]
        : input->const_data_ptr<int64_t>()[src];
    write_index(token_dtype, io.tokens.data, i, token);
    write_index(
        pos_dtype,
        io.positions.data,
        i,
        std::min<int64_t>(start + i, io.context_len - 1));
  }
  rearrange(io.past_len);
  fill_mask(io, start, n);
  io.valid = n;
  io.start = start;

  std::vector<EValue> inputs;
  inputs.reserve(3 + 2 * num_layers_ + cross_.size());
  inputs.emplace_back(*io.tokens.tensor);
  inputs.emplace_back(*io.mask.tensor);
  inputs.emplace_back(*io.positions.tensor);
  for (auto* buf : io.past_k) {
    inputs.emplace_back(*buf->tensor);
  }
  for (auto* buf : io.past_v) {
    inputs.emplace_back(*buf->tensor);
  }
  for (auto& buf : cross_) {
    inputs.emplace_back(*buf.tensor);
  }
  return inputs;
}

Error KvCacheIOManager::update(
    const std::vector<EValue>& model_outputs,
    const std::string& method) {
  auto& io = methods_.at(method);
  ET_CHECK_OR_RETURN_ERROR(
      model_outputs.size() == static_cast<size_t>(1 + 2 * num_layers_),
      InvalidState,
      "%s returned %zu outputs, expected %d.",
      method.c_str(),
      model_outputs.size(),
      1 + 2 * num_layers_);
  for (int32_t l = 0; l < num_layers_; ++l) {
    // K is [1, H, D, ar] -> past [1, H, D, P]: one strided write per (head,
    // dim) row.
    const auto& k_new = model_outputs[1 + l].toTensor();
    auto& k_past = *io.past_k[l];
    const size_t es = element_size(k_new.scalar_type());
    const size_t rows = k_new.numel() / io.ar_len;
    const std::byte* src = k_new.const_data_ptr<std::byte>();
    std::byte* dst = k_past.data;
    for (size_t r = 0; r < rows; ++r) {
      std::memcpy(
          dst + (r * io.past_len + io.start) * es,
          src + r * io.ar_len * es,
          io.valid * es);
    }
    // V is [1, H, ar, D] -> past [1, H, P, D]: one contiguous write per head.
    const auto& v_new = model_outputs[1 + num_layers_ + l].toTensor();
    auto& v_past = *io.past_v[l];
    const size_t heads = v_new.size(1);
    const size_t dim = v_new.size(3);
    const std::byte* vsrc = v_new.const_data_ptr<std::byte>();
    std::byte* vdst = v_past.data;
    for (size_t h = 0; h < heads; ++h) {
      std::memcpy(
          vdst + (h * io.past_len + io.start) * dim * es,
          vsrc + h * io.ar_len * dim * es,
          io.valid * dim * es);
    }
  }
  filled_ = io.start + io.valid;
  return Error::Ok;
}

Error KvCacheIOManager::load_encoder(const std::string& encoder_method) {
  ET_CHECK_OR_RETURN_ERROR(
      !cross_.empty(),
      InvalidState,
      "Load the decode method of a program with cross-attention inputs first.");
  ET_CHECK_OK_OR_RETURN_ERROR(module_.load_method(encoder_method));
  auto meta = module_.method_meta(encoder_method);
  ET_CHECK_OK_OR_RETURN_ERROR(meta.error());
  ET_CHECK_OR_RETURN_ERROR(
      meta->num_outputs() == cross_.size(),
      InvalidArgument,
      "%s returns %zu outputs; the decode step takes %zu cross-attention inputs.",
      encoder_method.c_str(),
      meta->num_outputs(),
      cross_.size());
  // The encoder writes straight into the decode step's inputs where it can.
  for (size_t i = 0; i < cross_.size(); ++i) {
    if (!meta->output_tensor_meta(i)->is_memory_planned()) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          module_.set_output(encoder_method, EValue(*cross_[i].tensor), i));
    }
  }
  return Error::Ok;
}

Error KvCacheIOManager::update_encoder(
    const std::vector<EValue>& encoder_outputs) {
  ET_CHECK_OR_RETURN_ERROR(
      encoder_outputs.size() == cross_.size(),
      InvalidArgument,
      "Got %zu encoder outputs for %zu cross-attention inputs.",
      encoder_outputs.size(),
      cross_.size());
  for (size_t i = 0; i < cross_.size(); ++i) {
    const auto& out = encoder_outputs[i].toTensor();
    if (out.const_data_ptr() != cross_[i].data) {
      ET_CHECK_OR_RETURN_ERROR(
          out.nbytes() == cross_[i].nbytes,
          InvalidArgument,
          "Encoder output %zu has %zu bytes, the input %zu.",
          i,
          out.nbytes(),
          cross_[i].nbytes);
      std::memcpy(cross_[i].data, out.const_data_ptr(), out.nbytes());
    }
  }
  return Error::Ok;
}

Result<std::vector<EValue>> KvCacheIOManager::prepare_prefill(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& prefill_method) {
  return prepare(input, start_pos, prefill_method);
}

Result<std::vector<EValue>> KvCacheIOManager::prepare_decode(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& decode_method) {
  return prepare(input, start_pos, decode_method);
}

Error KvCacheIOManager::update_prefill(
    const std::vector<EValue>& model_outputs,
    const std::string& prefill_method) {
  return update(model_outputs, prefill_method);
}

Error KvCacheIOManager::update_decode(
    const std::vector<EValue>& model_outputs,
    const std::string& decode_method) {
  return update(model_outputs, decode_method);
}

Result<executorch::aten::Tensor> KvCacheIOManager::logits(
    const std::vector<EValue>& model_outputs,
    const std::string& method) {
  auto& io = methods_.at(method);
  const auto& t = model_outputs[0].toTensor();
  const size_t vocab = logits_f_.size();
  // The row of the last real token (a prompt chunk may be padded).
  const size_t row = t.size(1) > 1 ? static_cast<size_t>(io.valid - 1) : 0;
  switch (t.scalar_type()) {
    case ScalarType::Float: {
      const float* src = t.const_data_ptr<float>() + row * vocab;
      std::copy(src, src + vocab, logits_f_.begin());
      break;
    }
    case ScalarType::UInt16: {
      const uint16_t* src = t.const_data_ptr<uint16_t>() + row * vocab;
      for (size_t i = 0; i < vocab; ++i) {
        logits_f_[i] = static_cast<float>(
            (static_cast<int64_t>(src[i]) - logits_zero_point_) *
            logits_scale_);
      }
      break;
    }
    case ScalarType::Byte: {
      const uint8_t* src = t.const_data_ptr<uint8_t>() + row * vocab;
      for (size_t i = 0; i < vocab; ++i) {
        logits_f_[i] = static_cast<float>(
            (static_cast<int64_t>(src[i]) - logits_zero_point_) *
            logits_scale_);
      }
      break;
    }
    default:
      ET_LOG(Error, "Unsupported logits dtype %d", (int)t.scalar_type());
      return Error::InvalidArgument;
  }
  return *logits_f_tensor_;
}

void register_kv_cache_io_manager() {
  static const bool registered = register_io_manager(
      KvCacheIOManager::kContract, [](ET_MODULE_NAMESPACE::Module& module) {
        return std::make_unique<KvCacheIOManager>(module);
      });
  (void)registered;
}

} // namespace llm
} // namespace extension
} // namespace executorch
