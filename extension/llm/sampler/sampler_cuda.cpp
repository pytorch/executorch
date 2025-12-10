#include <executorch/extension/llm/sampler/sampler_cuda.h>

#ifdef CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <ctime>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace executorch {
namespace extension {
namespace llm {

namespace {

bool is_cuda_pointer(const void* ptr) {
  if (ptr == nullptr) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  if (status != cudaSuccess) {
    return false;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice ||
      attributes.type == cudaMemoryTypeManaged;
#else
  return attributes.memoryType == cudaMemoryTypeDevice ||
      attributes.memoryType == cudaMemoryTypeManaged;
#endif
}

unsigned int random_u32(unsigned long long* state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long* state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

template <typename T>
struct TemperatureScaleExp {
  float max_val;
  float inv_temp;
  __host__ __device__ T operator()(T x) const {
    float scaled = static_cast<float>(x) * inv_temp - max_val;
    return static_cast<T>(expf(scaled));
  }
};

template <typename T>
struct ToFloat {
  __host__ __device__ float operator()(T x) const {
    return static_cast<float>(x);
  }
};

template <typename T>
struct NormalizeMul {
  float inv_sum;
  __host__ __device__ T operator()(T x) const {
    return static_cast<T>(static_cast<float>(x) * inv_sum);
  }
};

struct ProbGreater {
  __host__ __device__ bool operator()(
      const thrust::tuple<float, int32_t>& a,
      const thrust::tuple<float, int32_t>& b) const {
    return thrust::get<0>(a) > thrust::get<0>(b);
  }
};

template <typename T>
static void softmax_host(T* x, int size) {
  float max_val = static_cast<float>(x[0]);
  for (int i = 1; i < size; i++) {
    const float val = static_cast<float>(x[i]);
    if (val > max_val) {
      max_val = val;
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    float v = expf(static_cast<float>(x[i]) - max_val);
    x[i] = static_cast<T>(v);
    sum += v;
  }
  for (int i = 0; i < size; i++) {
    x[i] = static_cast<T>(static_cast<float>(x[i]) / sum);
  }
}

template <typename T>
struct ProbIndexHost {
  float prob;
  int32_t index;
};

} // namespace

CudaSampler::CudaSampler(
    int32_t vocab_size,
    float temperature,
    float topp,
    unsigned long long rng_seed,
    void* cuda_stream,
    SamplerWorkspace* workspace)
    : vocab_size_(vocab_size),
      inv_temperature_(
          static_cast<bool>(temperature) ? 1.0f / temperature : 0.0f),
      topp_(topp),
      rng_state_(rng_seed),
      cuda_stream_(cuda_stream),
      workspace_(workspace) {}

CudaSampler::CudaSampler(
    int32_t vocab_size,
    float temperature,
    void* cuda_stream,
    SamplerWorkspace* workspace)
    : vocab_size_(vocab_size),
      inv_temperature_(
          static_cast<bool>(temperature) ? 1.0f / temperature : 0.0f),
      topp_(kTopp),
      rng_state_(std::time(nullptr)),
      cuda_stream_(cuda_stream),
      workspace_(workspace) {}

template <typename T>
int32_t CudaSampler::sample(T* logits) {
  const bool use_cuda = is_cuda_pointer(logits);
  int32_t next = 0;
  if (!use_cuda) {
    if (inv_temperature_ == 0.0f) {
      next = sample_argmax_host(logits);
    } else {
      apply_temperature_and_softmax_host(logits);
      const float coin = random_f32(&rng_state_);
      if (topp_ <= 0 || topp_ >= 1) {
        next = sample_mult_host(logits, coin);
      } else {
        next = sample_topp_host(logits, coin);
      }
    }
    return next;
  }

  if (inv_temperature_ == 0.0f) {
    next = sample_argmax_cuda(logits);
  } else {
    apply_temperature_and_softmax_cuda(logits);
    const float coin = random_f32(&rng_state_);
    if (topp_ <= 0 || topp_ >= 1) {
      next = sample_mult_cuda(logits, coin);
    } else {
      next = sample_topp_cuda(logits, coin);
    }
  }
  return next;
}

template <typename T>
void CudaSampler::apply_temperature_and_softmax_host(T* logits) {
  for (int q = 0; q < vocab_size_; q++) {
    logits[q] = static_cast<T>(static_cast<float>(logits[q]) * inv_temperature_);
  }
  softmax_host(logits, vocab_size_);
}

template <typename T>
int32_t CudaSampler::sample_argmax_host(T* probabilities) {
  int max_i = 0;
  float max_p = static_cast<float>(probabilities[0]);
  for (int i = 1; i < vocab_size_; i++) {
    const float val = static_cast<float>(probabilities[i]);
    if (val > max_p) {
      max_i = i;
      max_p = val;
    }
  }
  return max_i;
}

template <typename T>
int32_t CudaSampler::sample_mult_host(T* probabilities, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < vocab_size_; i++) {
    cdf += static_cast<float>(probabilities[i]);
    if (coin < cdf) {
      return i;
    }
  }
  return vocab_size_ - 1;
}

template <typename T>
int32_t CudaSampler::sample_topp_host(T* probabilities, float coin) {
  int n = vocab_size_;
  int n0 = 0;
  std::unique_ptr<ProbIndexHost<T>[]> probindex =
      std::make_unique<ProbIndexHost<T>[]>(vocab_size_);

  const float cutoff = (1.0f - topp_) / (n - 1);
  for (int i = 0; i < n; i++) {
    const float prob = static_cast<float>(probabilities[i]);
    if (prob >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = prob;
      n0++;
    }
  }

  auto compare = [](const ProbIndexHost<T>& a, const ProbIndexHost<T>& b) {
    return a.prob > b.prob;
  };
  std::sort(probindex.get(), probindex.get() + n0, compare);

  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1;
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp_) {
      last_idx = i;
      break;
    }
  }

  const float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index;
}

template <typename T>
void CudaSampler::apply_temperature_and_softmax_cuda(T* logits) {
  thrust::device_ptr<T> logits_ptr(logits);
  const float max_val = thrust::transform_reduce(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      logits_ptr,
      logits_ptr + static_cast<size_t>(vocab_size_),
      ToFloat<T>{},
      -std::numeric_limits<float>::infinity(),
      thrust::maximum<float>());
  const float inv_temp = inv_temperature_;

  thrust::transform(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      logits_ptr,
      logits_ptr + static_cast<size_t>(vocab_size_),
      logits_ptr,
      TemperatureScaleExp<T>{max_val, inv_temp});

  const float sum = thrust::transform_reduce(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      logits_ptr,
      logits_ptr + static_cast<size_t>(vocab_size_),
      ToFloat<T>{},
      0.0f,
      thrust::plus<float>());
  if (sum <= 0.0f) {
    return;
  }
  const float inv_sum = 1.0f / sum;
  thrust::transform(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      logits_ptr,
      logits_ptr + static_cast<size_t>(vocab_size_),
      logits_ptr,
      NormalizeMul<T>{inv_sum});
}

template <typename T>
int32_t CudaSampler::sample_argmax_cuda(T* probabilities) {
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
  thrust::device_ptr<T> probs(probabilities);

  float* prob_data = nullptr;
  thrust::device_vector<float> local_probs;
  if (workspace_ != nullptr) {
    workspace_->reserve(static_cast<size_t>(vocab_size_));
    prob_data = workspace_->probs;
  } else {
    local_probs.resize(static_cast<size_t>(vocab_size_));
    prob_data = thrust::raw_pointer_cast(local_probs.data());
  }

  auto exec = thrust::cuda::par.on(stream);
  thrust::transform(
      exec,
      probs,
      probs + static_cast<size_t>(vocab_size_),
      prob_data,
      ToFloat<T>{});
  auto err = cudaGetLastError();
  ET_CHECK_MSG(err == cudaSuccess, "CUDA error: %s", cudaGetErrorString(err));

  using Pair = ::cub::KeyValuePair<int, float>;
  Pair* d_out = nullptr;
  Pair* local_out = nullptr;
  if (workspace_ != nullptr) {
    workspace_->ensure_argmax_capacity(1);
    d_out = static_cast<Pair*>(workspace_->argmax_out);
  } else {
    cudaMalloc(reinterpret_cast<void**>(&local_out), sizeof(Pair));
    d_out = local_out;
  }

  size_t temp_bytes = 0;
  cub::DeviceReduce::ArgMax(
      nullptr, temp_bytes, prob_data, d_out, vocab_size_, stream);

  uint8_t* temp_storage = nullptr;
  thrust::device_vector<uint8_t> local_temp;
  if (workspace_ != nullptr) {
    workspace_->ensure_temp_bytes(temp_bytes);
    temp_storage = workspace_->temp_storage;
  } else {
    local_temp.resize(temp_bytes);
    temp_storage = thrust::raw_pointer_cast(local_temp.data());
  }

  cub::DeviceReduce::ArgMax(
      temp_storage, temp_bytes, prob_data, d_out, vocab_size_, stream);

  Pair host_out{};
  cudaMemcpy(
      &host_out,
      d_out,
      sizeof(Pair),
      cudaMemcpyDeviceToHost);
  // cudaStreamSynchronize(stream);

  if (local_out != nullptr) {
    cudaFree(local_out);
  }
  return static_cast<int32_t>(host_out.key);
}

template <typename T>
int32_t CudaSampler::sample_mult_cuda(T* probabilities, float coin) {
  thrust::device_ptr<T> probs(probabilities);
  float* cdf_data = nullptr;
  thrust::device_vector<float> local_cdf;
  if (workspace_ != nullptr) {
    workspace_->reserve(static_cast<size_t>(vocab_size_));
    cdf_data = workspace_->probs;
  } else {
    local_cdf.resize(static_cast<size_t>(vocab_size_));
    cdf_data = thrust::raw_pointer_cast(local_cdf.data());
  }
  thrust::transform(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      probs,
      probs + static_cast<size_t>(vocab_size_),
      cdf_data,
      ToFloat<T>{});
  thrust::inclusive_scan(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      cdf_data,
      cdf_data + static_cast<size_t>(vocab_size_),
      cdf_data);
  auto it = thrust::upper_bound(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      cdf_data,
      cdf_data + static_cast<size_t>(vocab_size_),
      coin);
  if (it == cdf_data + static_cast<size_t>(vocab_size_)) {
    return vocab_size_ - 1;
  }
  return static_cast<int32_t>(it - cdf_data);
}

template <typename T>
int32_t CudaSampler::sample_topp_cuda(T* probabilities, float coin) {
  thrust::device_ptr<T> probs(probabilities);
  float* prob_data = nullptr;
  int32_t* index_data = nullptr;
  thrust::device_vector<float> local_prob_vec;
  thrust::device_vector<int32_t> local_indices;

  if (workspace_ != nullptr) {
    workspace_->reserve(static_cast<size_t>(vocab_size_));
    prob_data = workspace_->probs;
    index_data = workspace_->indices;
  } else {
    local_prob_vec.resize(static_cast<size_t>(vocab_size_));
    local_indices.resize(static_cast<size_t>(vocab_size_));
    prob_data = thrust::raw_pointer_cast(local_prob_vec.data());
    index_data = thrust::raw_pointer_cast(local_indices.data());
  }
  thrust::transform(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      probs,
      probs + static_cast<size_t>(vocab_size_),
      prob_data,
      ToFloat<T>{});
  thrust::sequence(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      index_data,
      index_data + static_cast<size_t>(vocab_size_));

  auto zipped_begin = thrust::make_zip_iterator(
      thrust::make_tuple(prob_data, index_data));
  auto zipped_end = zipped_begin + static_cast<size_t>(vocab_size_);

  thrust::sort(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      zipped_begin,
      zipped_end,
      ProbGreater{});

  thrust::inclusive_scan(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      prob_data,
      prob_data + static_cast<size_t>(vocab_size_),
      prob_data);
  auto cutoff_it = thrust::upper_bound(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      prob_data,
      prob_data + static_cast<size_t>(vocab_size_),
      topp_);
  const int last_idx = cutoff_it == prob_data + static_cast<size_t>(vocab_size_)
      ? static_cast<int>(vocab_size_ - 1)
      : static_cast<int>(cutoff_it - prob_data);
  const float cumulative_prob = prob_data[last_idx];
  const float target = coin * cumulative_prob;

  auto sample_it = thrust::upper_bound(
      thrust::cuda::par.on(static_cast<cudaStream_t>(cuda_stream_)),
      prob_data,
      prob_data + last_idx + 1,
      target);
  const int offset = sample_it == prob_data + last_idx + 1
      ? last_idx
      : static_cast<int>(sample_it - prob_data);
  return index_data[offset];
}

template int32_t CudaSampler::sample<float>(float*);
template int32_t CudaSampler::sample<uint16_t>(uint16_t*);
template int32_t CudaSampler::sample<executorch::aten::Half>(
    executorch::aten::Half*);
template int32_t CudaSampler::sample<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*);

template void CudaSampler::apply_temperature_and_softmax_cuda<float>(float*);
template void CudaSampler::apply_temperature_and_softmax_cuda<uint16_t>(
    uint16_t*);
template void
CudaSampler::apply_temperature_and_softmax_cuda<executorch::aten::Half>(
    executorch::aten::Half*);
template void
CudaSampler::apply_temperature_and_softmax_cuda<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*);

template int32_t CudaSampler::sample_argmax_cuda<float>(float*);
template int32_t CudaSampler::sample_argmax_cuda<uint16_t>(uint16_t*);
template int32_t CudaSampler::sample_argmax_cuda<executorch::aten::Half>(
    executorch::aten::Half*);
template int32_t CudaSampler::sample_argmax_cuda<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*);

template int32_t CudaSampler::sample_mult_cuda<float>(float*, float);
template int32_t CudaSampler::sample_mult_cuda<uint16_t>(uint16_t*, float);
template int32_t CudaSampler::sample_mult_cuda<executorch::aten::Half>(
    executorch::aten::Half*,
    float);
template int32_t CudaSampler::sample_mult_cuda<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*,
    float);

template int32_t CudaSampler::sample_topp_cuda<float>(float*, float);
template int32_t CudaSampler::sample_topp_cuda<uint16_t>(uint16_t*, float);
template int32_t CudaSampler::sample_topp_cuda<executorch::aten::Half>(
    executorch::aten::Half*,
    float);
template int32_t CudaSampler::sample_topp_cuda<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*,
    float);

template void CudaSampler::apply_temperature_and_softmax_host<float>(float*);
template void CudaSampler::apply_temperature_and_softmax_host<uint16_t>(
    uint16_t*);
template void
CudaSampler::apply_temperature_and_softmax_host<executorch::aten::Half>(
    executorch::aten::Half*);
template void
CudaSampler::apply_temperature_and_softmax_host<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*);

template int32_t CudaSampler::sample_argmax_host<float>(float*);
template int32_t CudaSampler::sample_argmax_host<uint16_t>(uint16_t*);
template int32_t CudaSampler::sample_argmax_host<executorch::aten::Half>(
    executorch::aten::Half*);
template int32_t CudaSampler::sample_argmax_host<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*);

template int32_t CudaSampler::sample_mult_host<float>(float*, float);
template int32_t CudaSampler::sample_mult_host<uint16_t>(uint16_t*, float);
template int32_t CudaSampler::sample_mult_host<executorch::aten::Half>(
    executorch::aten::Half*,
    float);
template int32_t CudaSampler::sample_mult_host<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*,
    float);

template int32_t CudaSampler::sample_topp_host<float>(float*, float);
template int32_t CudaSampler::sample_topp_host<uint16_t>(uint16_t*, float);
template int32_t CudaSampler::sample_topp_host<executorch::aten::Half>(
    executorch::aten::Half*,
    float);
template int32_t CudaSampler::sample_topp_host<executorch::aten::BFloat16>(
    executorch::aten::BFloat16*,
    float);

} // namespace llm
} // namespace extension
} // namespace executorch

#endif // CUDA_AVAILABLE
