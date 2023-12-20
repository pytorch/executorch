#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <xnnpack.h>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

struct ModelArgs {
  int32_t dim = 4096;
  int32_t n_layers = 32;
  int32_t n_heads = 32;
  int32_t n_kv_heads = 32;
  int32_t vocab_size = 32000;
  int32_t multiple_of = 256;
  float ffn_dim_multiplier;
  float norm_eps = 1e-5;
  int32_t max_batch_size = 32;
  int32_t max_seq_len = 256;
  int32_t kv_cache_size = 256;
};

enum class FullyConnectedOpType : uint8_t {
  QD8_F16_QC4W = 0,
  QD8_F16_QC8W,
  QD8_F32_QC4W,
  QD8_F32_QC8W,
};

enum class ComputeType : uint8_t {
  F16 = 0,
  F32,
  QuantizedInt8,
};

using torch::executor::ScalarType;
using torch::executor::testing::TensorFactory;

namespace {
class TensorFactoryWrapper {
 public:
  template <torch::executor::ScalarType DTYPE>
  static torch::executor::Tensor make_tensor(
      const std::vector<int32_t>& sizes) {
    sizes_vector_.push_back(sizes);
    auto& size = sizes_vector_.back();
    size[0] = size[0] * 4;
    return make_tensor_impl<DTYPE>(sizes);
  }

  template <torch::executor::ScalarType DTYPE>
  static torch::executor::Tensor make_tensor_impl(
      const std::vector<int32_t>& sizes) {
    ET_CHECK_MSG(
        false,
        "Cannot make tensor of type %s",
        torch::executor::toString(DTYPE));
  }

  static std::vector<std::vector<int32_t>> sizes_vector_;
  static TensorFactory<ScalarType::Char> tf_int8_;
  static TensorFactory<ScalarType::Byte> tf_uint8_;
  static TensorFactory<ScalarType::Int> tf_int_;
  static TensorFactory<ScalarType::Float> tf_float_;
  static TensorFactory<ScalarType::Short> tf_short_;
};
std::vector<std::vector<int32_t>> TensorFactoryWrapper::sizes_vector_;
TensorFactory<ScalarType::Char> TensorFactoryWrapper::tf_int8_;
TensorFactory<ScalarType::Byte> TensorFactoryWrapper::tf_uint8_;
TensorFactory<ScalarType::Int> TensorFactoryWrapper::tf_int_;
TensorFactory<ScalarType::Float> TensorFactoryWrapper::tf_float_;
TensorFactory<ScalarType::Short> TensorFactoryWrapper::tf_short_;

template <>
torch::executor::Tensor
TensorFactoryWrapper::make_tensor_impl<torch::executor::ScalarType::Char>(
    const std::vector<int32_t>& sizes) {
  return tf_int8_.ones(sizes);
}
template <>
torch::executor::Tensor
TensorFactoryWrapper::make_tensor_impl<torch::executor::ScalarType::Byte>(
    const std::vector<int32_t>& sizes) {
  return tf_uint8_.ones(sizes);
}
template <>
torch::executor::Tensor
TensorFactoryWrapper::make_tensor_impl<torch::executor::ScalarType::Int>(
    const std::vector<int32_t>& sizes) {
  return tf_int_.ones(sizes);
}
template <>
torch::executor::Tensor
TensorFactoryWrapper::make_tensor_impl<torch::executor::ScalarType::Float>(
    const std::vector<int32_t>& sizes) {
  return tf_float_.ones(sizes);
}
template <>
torch::executor::Tensor
TensorFactoryWrapper::make_tensor_impl<torch::executor::ScalarType::Half>(
    const std::vector<int32_t>& sizes) {
  return tf_short_.ones(sizes);
}

xnn_operator_t create_fully_connected(
    const FullyConnectedOpType fc_type,
    const int32_t input_channels,
    const int32_t output_channels) {
  xnn_operator_t op = nullptr;
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
    std::vector<uint8_t> kernel(output_channels * input_channels / 2, 1);
    std::vector<float> kernel_scales(output_channels, 1.0);
    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = +std::numeric_limits<float>::infinity();

    const xnn_status status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
        input_channels,
        output_channels,
        input_channels,
        output_channels,
        8,
        kernel_scales.data(),
        kernel.data(),
        nullptr,
        output_min,
        output_max,
        0,
        nullptr,
        nullptr,
        &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", fc_type);
}

xnn_operator_t create_sdpa(const ComputeType compute_type) {
  xnn_operator_t op = nullptr;
  if (compute_type == ComputeType::F16) {
    const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f16(
        xnn_attention_logits_cap_type_none, nullptr, 0, &op);
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", compute_type);
}

void run_fully_connected(
    const FullyConnectedOpType fc_type,
    xnn_operator_t op,
    const xnn_dynamic_quantization_params* qparams,
    const torch::executor::Tensor& input,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  int32_t batch_size = input.size(0);
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
    xnn_status status =
        xnn_reshape_fully_connected_nc_qd8_f16_qc4w(op, 1, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_fully_connected_nc_qd8_f16_qc4w(
        op,
        input.const_data_ptr<int8_t>(),
        output.mutable_data_ptr(),
        reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
            qparams));
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
  }
}

void run_sdpa(
    const ComputeType compute_type,
    xnn_operator_t op,
    const int32_t batch_size,
    const int32_t query_heads,
    const int32_t query_tokens,
    const int32_t kv_heads,
    const int32_t kv_tokens,
    const int32_t qk_channels,
    const int32_t v_channels,
    torch::executor::Tensor& query,
    torch::executor::Tensor& key,
    torch::executor::Tensor& value,
    torch::executor::Tensor& scale,
    torch::executor::Tensor& mask,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  if (compute_type == ComputeType::F16) {
    xnn_status status = xnn_reshape_scaled_dot_product_attention_nhtc_f16(
        op,
        batch_size,
        query_heads,
        query_tokens,
        kv_heads,
        kv_tokens,
        qk_channels,
        v_channels,
        &workspace_size,
        &workspace_alignment,
        threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success,
        "Failed to reshape sdpa xnnpack operator");

    constexpr std::ptrdiff_t alignment{64};
    // std::vector<char> workspace(workspace_size + alignment);
    // void* workspace_ptr = reinterpret_cast<void*>(workspace.data());
    void* workspace_ptr_orig = malloc(workspace_size + alignment);
    ET_CHECK_MSG(
        workspace_ptr_orig != nullptr, "Failed to allocate workspace memory");
    void* workspace_ptr = workspace_ptr_orig;
    status = xnn_setup_scaled_dot_product_attention_nhtc_f16(
        op,
        workspace_ptr,
        query.const_data_ptr<uint16_t>(),
        key.const_data_ptr<uint16_t>(),
        value.const_data_ptr<uint16_t>(),
        scale.const_data_ptr<uint16_t>(),
        mask.const_data_ptr<uint16_t>(),
        output.mutable_data_ptr<void>());
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup sdpa xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    free(workspace_ptr_orig);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", compute_type);
}
} // namespace

class MultiHeadedAttention {
 public:
  /*
  What is not accounted for:
  1. Dimension permute from NTHC to NHTC (for SDPA)
  2. Dimension permute from NHTC to NTHC for output projection
  3. RoPE
  */
  explicit MultiHeadedAttention(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : linear_type_(linear_type), sdpa_type_(sdpa_type) {
    ET_CHECK_MSG(
        args.n_heads == args.n_kv_heads,
        "MultiHeadedAttention only supports n_heads == n_kv_heads");
    int32_t input_channels = args.dim;
    int32_t output_channels = args.n_heads * (args.dim / args.n_heads);
    int32_t head_dim = args.dim / args.n_heads;

    q_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);
    k_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);
    v_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);
    o_proj_ =
        create_fully_connected(linear_type, output_channels, input_channels);

    // Not doing RoPE
    // SDPA
    sdpa_op_ = create_sdpa(sdpa_type);

    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      query_input_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Char>(
              {benchmarking_batch_size_, input_channels});
      query_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, output_channels});
      key_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, output_channels});
      value_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, output_channels});
      sdpa_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, output_channels});

      qparams_.resize(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
      std::generate(qparams_.begin(), qparams_.end(), [&]() {
        return xnn_dynamic_quantization_params{0, 1.f};
      });

      k_cache_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_,
               args.n_heads,
               args.max_seq_len,
               head_dim});
      v_cache_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_,
               args.n_heads,
               args.max_seq_len,
               head_dim});
      scales_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {head_dim});
      mask_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {1, args.max_seq_len});

      query_heads_ = args.n_heads;
      kv_heads_ = args.n_heads;
      query_tokens_ = 1;
      kv_tokens_ = args.max_seq_len;
      qk_channels_ = head_dim;
      v_channels_ = head_dim;
      return;
    }
    ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
  }

  MultiHeadedAttention(const MultiHeadedAttention& other) =
      delete; // Copy constructor
  MultiHeadedAttention& operator=(const MultiHeadedAttention& other) =
      delete; // Assignment operator
  MultiHeadedAttention(MultiHeadedAttention&& other) noexcept =
      delete; // Move constructor
  MultiHeadedAttention& operator=(MultiHeadedAttention&& other) noexcept =
      delete; // Move assignment operator

  ~MultiHeadedAttention() {
    xnn_delete_operator(q_proj_);
    xnn_delete_operator(k_proj_);
    xnn_delete_operator(v_proj_);
    xnn_delete_operator(o_proj_);
    xnn_delete_operator(sdpa_op_);
  }

  void run(
      const torch::executor::Tensor& q,
      const torch::executor::Tensor& k_cache,
      const torch::executor::Tensor& v_cache) {
    // Fake it
  }

  void run_bench() {
    run_fully_connected(
        linear_type_,
        q_proj_,
        qparams_.data(),
        query_input_.value(),
        query_output_.value());
    run_fully_connected(
        linear_type_,
        k_proj_,
        qparams_.data(),
        query_input_.value(),
        key_output_.value());
    run_fully_connected(
        linear_type_,
        v_proj_,
        qparams_.data(),
        query_input_.value(),
        value_output_.value());
    // Missing
    // - apply rotary embedding on q and k
    run_sdpa(
        sdpa_type_,
        sdpa_op_,
        benchmarking_batch_size_,
        query_heads_,
        query_tokens_,
        kv_heads_,
        kv_tokens_,
        qk_channels_,
        v_channels_,
        query_output_.value(),
        k_cache_.value(),
        v_cache_.value(),
        scales_.value(),
        mask_.value(),
        sdpa_output_.value());
    run_fully_connected(
        linear_type_,
        o_proj_,
        qparams_.data(),
        sdpa_output_.value(),
        value_output_.value());
  }

 private:
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  ComputeType sdpa_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> query_input_, k_cache_, v_cache_;
  exec_aten::optional<torch::executor::Tensor> scales_, mask_;
  exec_aten::optional<torch::executor::Tensor> sdpa_output_;
  exec_aten::optional<torch::executor::Tensor> query_output_, key_output_,
      value_output_, output_;
  xnn_operator_t q_proj_;
  xnn_operator_t k_proj_;
  xnn_operator_t v_proj_;
  xnn_operator_t o_proj_;
  xnn_operator_t sdpa_op_;
  int32_t query_heads_, kv_heads_, query_tokens_, kv_tokens_, qk_channels_,
      v_channels_;
};

class FeedForward {
 public:
  /*
  What is not accounted for:
  1. Silu
  */
  explicit FeedForward(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type)
      : linear_type_(linear_type) {
    int32_t hidden_dim = 4 * args.dim;
    int32_t n_hidden = 2 * hidden_dim / 3;
    int32_t mask = ~(args.multiple_of - 1);
    int32_t intermediate_size = (n_hidden + args.multiple_of - 1) & mask;

    w1_ = create_fully_connected(linear_type, args.dim, intermediate_size);
    w3_ = create_fully_connected(linear_type, args.dim, intermediate_size);
    w2_ = create_fully_connected(linear_type, intermediate_size, args.dim);

    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      ffw_input_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Char>(
              {benchmarking_batch_size_, args.dim});
      w1_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, intermediate_size});
      w3_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, intermediate_size});
      w2_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, args.dim});

      qparams_.resize(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
      std::generate(qparams_.begin(), qparams_.end(), [&]() {
        return xnn_dynamic_quantization_params{0, 1.f};
      });
      return;
    }
    ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
  }

  FeedForward(const FeedForward& other) = delete;
  FeedForward& operator=(const FeedForward& other) = delete;
  FeedForward(FeedForward&& other) = delete; // Move constructor
  FeedForward& operator=(FeedForward&& other) noexcept =
      delete; // Move assignment operator

  ~FeedForward() {
    xnn_delete_operator(w1_);
    xnn_delete_operator(w2_);
    xnn_delete_operator(w3_);
    // xnn_delete_operator(silu_);
  }

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    run_fully_connected(
        linear_type_,
        w1_,
        qparams_.data(),
        ffw_input_.value(),
        w1_output_.value());
    run_fully_connected(
        linear_type_,
        w3_,
        qparams_.data(),
        ffw_input_.value(),
        w3_output_.value());
    run_fully_connected(
        linear_type_,
        w2_,
        qparams_.data(),
        w3_output_.value(),
        w2_output_.value());
  }

 private:
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> ffw_input_, w1_output_,
      w3_output_, w2_output_;
  xnn_operator_t w1_;
  xnn_operator_t w2_;
  xnn_operator_t w3_;
  xnn_operator_t silu_;
};

class TransformerBlock {
 public:
  /*
  What is not accounted for:
  1. Attention RMSNorm
  2. FFN RMSNorm
  */
  explicit TransformerBlock(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : multi_headed_attention_(args, linear_type, sdpa_type),
        feedforward_(args, linear_type) {}

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    multi_headed_attention_.run_bench();
    feedforward_.run_bench();
  }

 private:
  MultiHeadedAttention multi_headed_attention_;
  FeedForward feedforward_;
};

class Transformer {
 public:
  /*
  What is not accounted for:
  1. Generating and selecting input pos dependent causal mask
  2. frequ_cis populating for RoPE
  In run_bench
  1. Token embeddings
  2. RMSNorm on transformer blocks
  */
  explicit Transformer(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : linear_type_(linear_type) {
    transformer_blocks_.reserve(args.n_layers);
    for (int i = 0; i < args.n_layers; ++i) {
      transformer_blocks_.emplace_back(
          std::make_unique<TransformerBlock>(args, linear_type, sdpa_type));
    }

    // Not sure if we should quantize the last linear layer or not. For now
    // assuming we do.
    out_logits_ =
        create_fully_connected(linear_type, args.dim, args.vocab_size);

    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      out_logits_input_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Char>(
              {benchmarking_batch_size_, args.dim});
      out_logits_output_ =
          TensorFactoryWrapper::make_tensor<torch::executor::ScalarType::Half>(
              {benchmarking_batch_size_, args.vocab_size});

      qparams_.resize(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
      std::generate(qparams_.begin(), qparams_.end(), [&]() {
        return xnn_dynamic_quantization_params{0, 1.f};
      });
      return;
    }
    ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
  }

  Transformer(const Transformer& other) = delete;
  Transformer& operator=(const Transformer& other) = delete;
  Transformer(Transformer&& other) = delete; // Move constructor
  Transformer& operator=(Transformer&& other) noexcept =
      delete; // Move assignment operator

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    for (auto& transformer_block : transformer_blocks_) {
      transformer_block->run_bench();
    }
    run_fully_connected(
        linear_type_,
        out_logits_,
        qparams_.data(),
        out_logits_input_.value(),
        out_logits_output_.value());
  }

 private:
  std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;
  xnn_operator_t out_logits_;
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> out_logits_input_,
      out_logits_output_;
};

static void benchmark_llama2_7b() {
  ModelArgs args;
  const int32_t kIterations = 200;
  // Need to benchmark pre-fill separately.
  Transformer transformer(
      args, FullyConnectedOpType::QD8_F16_QC4W, ComputeType::F16);
  auto start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    transformer.run_bench();
  }
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
  std::cout << "Elapsed time: " << elapsed_us << " microseconds" << std::endl;
  std::cout << "Elapsed time per iter(" << kIterations
            << "): " << elapsed_us / kIterations << " microseconds"
            << std::endl;
}

// BENCHMARK(benchmark_llama2_7b)->Unit(benchmark::kMicrosecond)->UseRealTime();
int main(int argc, char** argv) {
  torch::executor::runtime_init();
  xnn_status status = xnn_initialize(/*allocator =*/nullptr);
  ET_CHECK_MSG(status == xnn_status_success, "failed to initialize xnnpack");
  benchmark_llama2_7b();
}
