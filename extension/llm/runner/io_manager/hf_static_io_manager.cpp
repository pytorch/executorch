/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/io_manager/hf_static_io_manager.h>

#include <executorch/extension/llm/runner/constants.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace executorch {
namespace extension {
namespace llm {

using ::executorch::aten::ScalarType;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using ::executorch::runtime::TensorInfo;

namespace {
constexpr const char* kForward = "forward";
constexpr size_t kAlignment = 64;

size_t align_up(size_t n) {
  return (n + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace

HfStaticIOManager::HfStaticIOManager(ET_MODULE_NAMESPACE::Module& module)
    : IOManager(module), module_(module) {
  if (auto& factory = io_buffer_allocator_factory()) {
    allocator_ = factory();
  }
}

HfStaticIOManager::~HfStaticIOManager() {
  methods_.clear();
  caches_.clear();
  for (const auto& block : blocks_) {
    if (block.from_allocator) {
      allocator_->deallocate(block.data);
    } else {
      ::operator delete(block.data, std::align_val_t(kAlignment));
    }
  }
}

std::pair<std::string, std::string> HfStaticIOManager::method_names(
    const std::unordered_set<std::string>& available) const {
  if (available.count(kDecodeMethod)) {
    return std::make_pair(
        available.count(kPrefillMethod) ? kPrefillMethod : kDecodeMethod,
        kDecodeMethod);
  }
  return available.count(kForward) ? std::make_pair(kForward, kForward)
                                   : std::make_pair("", "");
}

int64_t HfStaticIOManager::max_tokens_per_step(
    const std::string& method) const {
  auto it = methods_.find(method);
  return it == methods_.end() ? 1 : it->second.query_len;
}

int64_t HfStaticIOManager::max_tokens_per_step(
    const std::string& method,
    int64_t start_pos) const {
  auto it = methods_.find(method);
  if (it == methods_.end() || start_pos < 0 ||
      start_pos > context_len_ - it->second.query_len) {
    return 0;
  }
  return it->second.query_len;
}

void HfStaticIOManager::allocate(
    const std::vector<std::pair<Buffer*, TensorInfo>>& buffers) {
  // One block for all of them: a delegate registers shared memory per block.
  size_t total = 0;
  for (const auto& [buf, info] : buffers) {
    total = align_up(total) + info.nbytes();
  }
  void* block = allocator_ ? allocator_->allocate(total, kAlignment) : nullptr;
  const bool from_allocator = block != nullptr;
  if (!from_allocator) {
    block = ::operator new(total, std::align_val_t(kAlignment));
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

Error HfStaticIOManager::load_metadata() {
  if (metadata_loaded_) {
    return Error::Ok;
  }
  auto names = module_.method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(names.error());
  for (const auto* name :
       {"get_io_contract_version",
        "get_cache_dtype",
        "get_n_layers",
        "get_max_seq_len"}) {
    ET_CHECK_OR_RETURN_ERROR(
        names->count(name),
        InvalidArgument,
        "hf_static: required metadata method %s is missing.",
        name);
  }
  auto version = module_.get("get_io_contract_version");
  ET_CHECK_OK_OR_RETURN_ERROR(version.error());
  ET_CHECK_OR_RETURN_ERROR(
      version->isInt() && version->toInt() >= 1,
      InvalidArgument,
      "hf_static: get_io_contract_version must be a positive integer.");
  ET_CHECK_OR_RETURN_ERROR(
      version->toInt() <= kContractVersion,
      NotSupported,
      "hf_static: contract version %" PRId64
      " is newer than supported version %" PRId64 ".",
      version->toInt(),
      kContractVersion);
  auto dtype = module_.get("get_cache_dtype");
  ET_CHECK_OK_OR_RETURN_ERROR(dtype.error());
  ET_CHECK_OR_RETURN_ERROR(
      dtype->isString(),
      InvalidArgument,
      "hf_static: get_cache_dtype must be float32 or float16.");
  const std::string label(dtype->toString());
  ET_CHECK_OR_RETURN_ERROR(
      label == "float32" || label == "float16",
      InvalidArgument,
      "hf_static: unsupported cache dtype '%s'.",
      label.c_str());
  auto layers = module_.get("get_n_layers");
  ET_CHECK_OK_OR_RETURN_ERROR(layers.error());
  auto context = module_.get("get_max_seq_len");
  ET_CHECK_OK_OR_RETURN_ERROR(context.error());
  ET_CHECK_OR_RETURN_ERROR(
      layers->isInt() && layers->toInt() > 0 &&
          layers->toInt() <= std::numeric_limits<int32_t>::max() &&
          context->isInt() && context->toInt() > 0 &&
          context->toInt() <= std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      "hf_static: get_n_layers and get_max_seq_len must be positive integers.");
  cache_dtype_ = label == "float32" ? ScalarType::Float : ScalarType::Half;
  num_layers_ = static_cast<int32_t>(layers->toInt());
  context_len_ = context->toInt();
  metadata_loaded_ = true;
  return Error::Ok;
}

Error HfStaticIOManager::load_method(const std::string& method) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_metadata());
  if (methods_.count(method)) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module_.load_method(method));
  auto meta_result = module_.method_meta(method);
  ET_CHECK_OK_OR_RETURN_ERROR(meta_result.error());
  auto meta = std::move(*meta_result);
  Method io;
  io.num_inputs = meta.num_inputs();
  std::vector<TensorInfo> infos;
  for (size_t i = 0; i < io.num_inputs; ++i) {
    auto info = meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(info.error());
    infos.push_back(*info);
  }
  auto is = [&](size_t i, ScalarType dtype, size_t rank) {
    return i < infos.size() && infos[i].scalar_type() == dtype &&
        infos[i].sizes().size() == rank;
  };
  // input_ids, the attention mask(s), position_ids, then per layer keys,
  // values and the fill counter.
  size_t i = 0;
  ET_CHECK_OR_RETURN_ERROR(
      is(i, ScalarType::Long, 2),
      InvalidArgument,
      "hf_static: %s: input 0 must be int64 input_ids [1, N].",
      method.c_str());
  io.input_ids = i++;
  const auto token_sizes = infos[io.input_ids].sizes();
  ET_CHECK_OR_RETURN_ERROR(
      token_sizes[0] == 1 && token_sizes[1] > 0,
      InvalidArgument,
      "hf_static: %s: input_ids must be [1, N] with N > 0.",
      method.c_str());
  io.query_len = token_sizes[1];
  while (is(i, ScalarType::Bool, 4)) {
    const auto sizes = infos[i].sizes();
    ET_CHECK_OR_RETURN_ERROR(
        sizes[0] == 1 && sizes[1] == 1 && sizes[2] == io.query_len &&
            sizes[3] > 0 && sizes[3] <= context_len_,
        InvalidArgument,
        "hf_static: %s: mask %zu must be [1, 1, N, W], 0 < W <= context.",
        method.c_str(),
        i);
    io.masks.push_back(i++);
  }
  ET_CHECK_OR_RETURN_ERROR(
      !io.masks.empty() && is(i, ScalarType::Long, 2),
      InvalidArgument,
      "hf_static: %s: expected bool attention masks, then int64 position_ids.",
      method.c_str());
  io.position_ids = i++;
  const auto position_sizes = infos[io.position_ids].sizes();
  ET_CHECK_OR_RETURN_ERROR(
      position_sizes[0] == 1 && position_sizes[1] == io.query_len,
      InvalidArgument,
      "hf_static: %s: position_ids must be [1, N].",
      method.c_str());
  io.first_cache = i;
  const size_t num_cache_inputs = io.num_inputs - io.first_cache;
  ET_CHECK_OR_RETURN_ERROR(
      num_cache_inputs == 3 * static_cast<size_t>(num_layers_),
      InvalidArgument,
      "hf_static: %s: expected three cache inputs per layer, matching get_n_layers.",
      method.c_str());
  for (size_t c = io.first_cache; c < io.num_inputs; c += 3) {
    ET_CHECK_OR_RETURN_ERROR(
        is(c, cache_dtype_, 4) && is(c + 1, cache_dtype_, 4) &&
            is(c + 2, ScalarType::Long, 0),
        InvalidArgument,
        "hf_static: %s: inputs %zu..%zu must be float keys/values matching get_cache_dtype and an int64 counter.",
        method.c_str(),
        c,
        c + 2);
    const auto keys = infos[c].sizes();
    const auto values = infos[c + 1].sizes();
    ET_CHECK_OR_RETURN_ERROR(
        keys[0] == 1 && keys[1] > 0 && keys[2] >= io.query_len &&
            keys[2] <= context_len_ && keys[3] > 0 &&
            std::equal(keys.begin(), keys.end(), values.begin()),
        InvalidArgument,
        "hf_static: %s: keys and values must share [1, H, L, D], N <= L <= context.",
        method.c_str());
  }
  for (size_t s = 0; s < io.first_cache; ++s) {
    io.step_inputs[s]; // Allocated once the method is stored, below.
  }
  ET_CHECK_OR_RETURN_ERROR(
      meta.num_outputs() == num_cache_inputs + 1 ||
          meta.num_outputs() == 2 * num_cache_inputs + 1,
      InvalidArgument,
      "hf_static: %s must return mutated inputs, logits, and optionally the user cache state.",
      method.c_str());
  if (meta.num_outputs() == 2 * num_cache_inputs + 1) {
    for (size_t c = 0; c < num_cache_inputs; ++c) {
      auto state = meta.output_tensor_meta(num_cache_inputs + 1 + c);
      ET_CHECK_OK_OR_RETURN_ERROR(state.error());
      const auto& input = infos[io.first_cache + c];
      ET_CHECK_OR_RETURN_ERROR(
          state->scalar_type() == input.scalar_type() &&
              state->sizes().size() == input.sizes().size() &&
              std::equal(
                  input.sizes().begin(),
                  input.sizes().end(),
                  state->sizes().begin()),
          InvalidArgument,
          "hf_static: %s: user state output %zu differs from its input.",
          method.c_str(),
          c);
    }
  }
  const auto logits_meta = meta.output_tensor_meta(num_cache_inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(logits_meta.error());
  const auto logits_sizes = logits_meta->sizes();
  ET_CHECK_OR_RETURN_ERROR(
      (logits_meta->scalar_type() == ScalarType::Float ||
       logits_meta->scalar_type() == ScalarType::Half) &&
          logits_sizes.size() == 3 && logits_sizes[0] == 1 &&
          logits_sizes[1] == io.query_len && logits_sizes[2] > 0,
      InvalidArgument,
      "hf_static: %s: logits must be float32/float16 [1, N, V].",
      method.c_str());
  if (caches_.empty()) {
    caches_.resize(num_cache_inputs);
    std::vector<std::pair<Buffer*, TensorInfo>> cache_buffers;
    for (size_t c = 0; c < num_cache_inputs; ++c) {
      cache_buffers.emplace_back(&caches_[c], infos[io.first_cache + c]);
    }
    allocate(cache_buffers);
    logits_f_.assign(logits_sizes[2], 0.0f);
    logits_f_tensor_ = from_blob(
        logits_f_.data(),
        {1, 1, static_cast<executorch::aten::SizesType>(logits_f_.size())},
        ScalarType::Float);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        num_cache_inputs == caches_.size() &&
            static_cast<size_t>(logits_sizes[2]) == logits_f_.size(),
        InvalidArgument,
        "hf_static: %s takes a different cache or vocabulary than the other method.",
        method.c_str());
    for (size_t c = 0; c < caches_.size(); ++c) {
      const auto& info = infos[io.first_cache + c];
      const auto& tensor = *caches_[c].tensor;
      ET_CHECK_OR_RETURN_ERROR(
          info.scalar_type() == tensor.scalar_type() &&
              info.sizes().size() == tensor.sizes().size() &&
              std::equal(
                  info.sizes().begin(),
                  info.sizes().end(),
                  tensor.sizes().begin()),
          InvalidArgument,
          "hf_static: %s: cache input %zu differs from the other method's dtype or shape.",
          method.c_str(),
          c);
    }
  }
  // Every other output is an input written in place. Unplanned logits are
  // written into a buffer of ours (shared memory, so a delegate writes them
  // directly); planned ones come back in the program's own memory.
  const bool bind_logits = !logits_meta->is_memory_planned();
  auto& stored = methods_.emplace(method, std::move(io)).first->second;
  std::vector<std::pair<Buffer*, TensorInfo>> method_buffers;
  for (auto& [s, buf] : stored.step_inputs) {
    method_buffers.emplace_back(&buf, infos[s]);
  }
  if (bind_logits) {
    method_buffers.emplace_back(&stored.logits, *logits_meta);
  }
  allocate(method_buffers);
  if (meta.num_outputs() == 2 * num_cache_inputs + 1) {
    for (size_t c = 0; c < num_cache_inputs; ++c) {
      const size_t output = num_cache_inputs + 1 + c;
      if (!meta.output_tensor_meta(output)->is_memory_planned()) {
        ET_CHECK_OK_OR_RETURN_ERROR(
            module_.set_output(method, EValue(*caches_[c].tensor), output));
      }
    }
  }
  if (bind_logits) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_.set_output(
        method, EValue(*stored.logits.tensor), num_cache_inputs));
  }
  return Error::Ok;
}

Error HfStaticIOManager::load(
    const std::string& prefill_method,
    const std::string& decode_method) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(prefill_method));
  return load_method(decode_method);
}

Error HfStaticIOManager::reset(
    const std::string& prefill_method,
    const std::string& decode_method) {
  ET_CHECK_OK_OR_RETURN_ERROR(load(prefill_method, decode_method));
  for (auto& buf : caches_) {
    std::memset(buf.data, 0, buf.nbytes);
  }
  return Error::Ok;
}

Result<std::vector<EValue>> HfStaticIOManager::prepare(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& method) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method));
  auto& io = methods_.at(method);
  ET_CHECK_OR_RETURN_ERROR(
      start_pos && input && start_pos->scalar_type() == ScalarType::Long &&
          start_pos->numel() == 1 &&
          (input->scalar_type() == ScalarType::Long ||
           input->scalar_type() == ScalarType::Int),
      InvalidArgument,
      "hf_static: expected int64 start_pos and int32/int64 tokens.");
  const int64_t n = input->numel();
  const int64_t start = start_pos->const_data_ptr<int64_t>()[0];
  ET_CHECK_OR_RETURN_ERROR(
      start >= 0,
      InvalidArgument,
      "hf_static: negative position %" PRId64 ".",
      start);
  ET_CHECK_OR_RETURN_ERROR(
      n >= 1 && n <= io.query_len,
      InvalidArgument,
      "hf_static: %s takes at most %" PRId64 " tokens per step, got %" PRId64
      ".",
      method.c_str(),
      io.query_len,
      n);
  // The step writes all its rows, padding included.
  ET_CHECK_OR_RETURN_ERROR(
      start <= context_len_ - io.query_len,
      OutOfResources,
      "hf_static: %s at position %" PRId64
      " would write past the cache length %" PRId64 ".",
      method.c_str(),
      start,
      context_len_);
  // A short chunk is padded with its last token, at the following positions;
  // the rows it writes past the real tokens are rewritten before they're read.
  auto* ids = static_cast<int64_t*>(io.step_inputs.at(io.input_ids).data);
  auto* pos = static_cast<int64_t*>(io.step_inputs.at(io.position_ids).data);
  for (int64_t i = 0; i < io.query_len; ++i) {
    const int64_t src = std::min(i, n - 1);
    ids[i] = input->scalar_type() == ScalarType::Int
        ? input->const_data_ptr<int32_t>()[src]
        : input->const_data_ptr<int64_t>()[src];
    pos[i] = std::min(start + i, context_len_ - 1);
  }
  // Row i sees slots 0..start + i, as HF's generator builds the mask. A
  // sliding window's (window-sized) mask also tells real rows from padding,
  // which sees nothing: the program shifts the window by the real rows.
  for (size_t m : io.masks) {
    auto& buf = io.step_inputs.at(m);
    const auto sizes = buf.tensor->sizes();
    const int64_t width = sizes[sizes.size() - 1];
    auto* mask = static_cast<bool*>(buf.data);
    for (int64_t i = 0; i < io.query_len; ++i) {
      const bool padding = i >= n && width < context_len_;
      const int64_t visible = padding ? 0 : std::min(width, start + i + 1);
      std::memset(mask + i * width, 1, visible);
      std::memset(mask + i * width + visible, 0, width - visible);
    }
  }
  for (size_t c = 2; c < caches_.size(); c += 3) {
    *static_cast<int64_t*>(caches_[c].data) = start;
  }
  io.valid = n;
  std::vector<EValue> values;
  values.reserve(io.num_inputs);
  for (size_t i = 0; i < io.num_inputs; ++i) {
    values.emplace_back(
        i < io.first_cache ? *io.step_inputs.at(i).tensor
                           : *caches_[i - io.first_cache].tensor);
  }
  return values;
}

Result<std::vector<EValue>> HfStaticIOManager::prepare_prefill(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& prefill_method) {
  return prepare(input, start_pos, prefill_method);
}

Result<std::vector<EValue>> HfStaticIOManager::prepare_decode(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& decode_method) {
  return prepare(input, start_pos, decode_method);
}

Error HfStaticIOManager::update_prefill(
    const std::vector<EValue>& model_outputs,
    const std::string& prefill_method) {
  (void)model_outputs;
  (void)prefill_method;
  return Error::Ok;
}

Error HfStaticIOManager::update_decode(
    const std::vector<EValue>& model_outputs,
    const std::string& decode_method) {
  (void)model_outputs;
  (void)decode_method;
  return Error::Ok;
}

Result<executorch::aten::Tensor> HfStaticIOManager::logits(
    const std::vector<EValue>& model_outputs,
    const std::string& method) {
  // The program returns its mutated inputs before its own outputs.
  const size_t index = static_cast<size_t>(3 * num_layers_);
  ET_CHECK_OR_RETURN_ERROR(
      model_outputs.size() > index && model_outputs[index].isTensor(),
      InvalidState,
      "hf_static: expected the logits at output %zu of %zu.",
      index,
      model_outputs.size());
  const auto& t = model_outputs[index].toTensor();
  const auto& io = methods_.at(method);
  const size_t vocab = logits_f_.size();
  // The row of the last real token (a prompt chunk may be padded).
  const size_t row = t.dim() == 3 && t.size(1) > 1 ? io.valid - 1 : 0;
  switch (t.scalar_type()) {
    case ScalarType::Float: {
      const float* src = t.const_data_ptr<float>() + row * vocab;
      std::copy(src, src + vocab, logits_f_.begin());
      break;
    }
    case ScalarType::Half: {
      const auto* src =
          t.const_data_ptr<executorch::aten::Half>() + row * vocab;
      for (size_t i = 0; i < vocab; ++i) {
        logits_f_[i] = static_cast<float>(src[i]);
      }
      break;
    }
    default:
      ET_LOG(
          Error,
          "hf_static: unsupported logits dtype %d",
          (int)t.scalar_type());
      return Error::InvalidArgument;
  }
  return *logits_f_tensor_;
}

void register_hf_static_io_manager() {
  static const bool registered = register_io_manager(
      HfStaticIOManager::kContract, [](ET_MODULE_NAMESPACE::Module& module) {
        return std::make_unique<HfStaticIOManager>(module);
      });
  (void)registered;
}

} // namespace llm
} // namespace extension
} // namespace executorch
