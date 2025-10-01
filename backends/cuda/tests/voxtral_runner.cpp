#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/tensor.h>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::extension::module::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

std::vector<executorch::aten::SizesType> to_sizes(
    std::initializer_list<int64_t> dims) {
  return std::vector<executorch::aten::SizesType>(dims.begin(), dims.end());
}

std::string format_shape(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "[";
  const auto& sizes = tensor.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << sizes[i];
  }
  oss << "]";
  return oss.str();
}

void print_tensor_summary(const std::string& label, const Tensor& tensor) {
  std::cout << "    " << label
            << ": dtype=" << executorch::runtime::toString(tensor.scalar_type())
            << ", shape=" << format_shape(tensor)
            << ", numel=" << tensor.numel() << std::endl;
}

TensorPtr create_audio_input() {
  const auto sizes = to_sizes({3, 128, 3000});
  const size_t numel = 3ull * 128ull * 3000ull;
  std::vector<float> data(numel, 0.5f);
  return make_tensor_ptr<float>(
      sizes, std::move(data), {}, {}, ScalarType::BFloat16);
}

TensorPtr create_token_ids_input() {
  const auto sizes = to_sizes({1, 1138});
  std::vector<int64_t> data(static_cast<size_t>(1) * 1138, 0);
  return make_tensor_ptr<int64_t>(sizes, std::move(data));
}

TensorPtr create_positions_input() {
  const auto sizes = to_sizes({1138});
  std::vector<int64_t> data(static_cast<size_t>(1138), 0);
  return make_tensor_ptr<int64_t>(sizes, std::move(data));
}

TensorPtr create_fallback_text_embedding() {
  const auto sizes = to_sizes({1, 1138, 3072});
  const size_t numel = 1ull * 1138ull * 3072ull;
  std::vector<float> data(numel, 0.0f);
  return make_tensor_ptr<float>(
      sizes, std::move(data), {}, {}, ScalarType::BFloat16);
}

struct MethodTiming {
  double load_ms{0.0};
  double run_ms{0.0};
};

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <path/to/model.pte> <path/to/aoti_cuda_blob.ptd>"
              << std::endl;
    return 1;
  }

  const std::string program_path = argv[1];
  const std::string data_map_path = argv[2];

  try {
    Module module(program_path, data_map_path);

    const auto program_load_start = Clock::now();
    const Error program_load_error = module.load();
    const auto program_load_end = Clock::now();
    if (program_load_error != Error::Ok) {
      std::cerr << "Failed to load ExecuTorch program: error code "
                << static_cast<int>(program_load_error) << std::endl;
      return 1;
    }
    const DurationMs program_load_latency =
        program_load_end - program_load_start;

    MethodTiming audio_timing;
    MethodTiming token_timing;
    MethodTiming text_timing;

    auto measure_method_load =
        [&](const std::string& name) -> std::pair<Error, double> {
      const auto start = Clock::now();
      const Error err = module.load_method(name);
      const auto end = Clock::now();
      return {err, DurationMs(end - start).count()};
    };

    // audio_encoder
    {
      const auto [err, load_ms] = measure_method_load("audio_encoder");
      if (err != Error::Ok) {
        std::cerr << "Failed to load method audio_encoder: error code "
                  << static_cast<int>(err) << std::endl;
        return 1;
      }
      audio_timing.load_ms = load_ms;

      const TensorPtr audio_input = create_audio_input();
      std::vector<EValue> inputs;
      inputs.emplace_back(audio_input);

      const auto run_start = Clock::now();
      Result<std::vector<EValue>> output_result =
          module.execute("audio_encoder", inputs);
      const auto run_end = Clock::now();
      audio_timing.run_ms = DurationMs(run_end - run_start).count();

      if (output_result.error() != Error::Ok) {
        std::cerr << "audio_encoder execution failed: error code "
                  << static_cast<int>(output_result.error()) << std::endl;
        return 1;
      }

      const auto& outputs = output_result.get();
      if (!outputs.empty() && outputs[0].isTensor()) {
        print_tensor_summary("audio_encoder output", outputs[0].toTensor());
      }
    }

    EValue token_output;
    bool token_executed = false;

    // token_embedding
    {
      const auto [err, load_ms] = measure_method_load("token_embedding");
      if (err != Error::Ok) {
        std::cerr << "Failed to load method token_embedding: error code "
                  << static_cast<int>(err) << std::endl;
        return 1;
      }
      token_timing.load_ms = load_ms;

      const TensorPtr token_ids = create_token_ids_input();
      std::vector<EValue> inputs;
      inputs.emplace_back(token_ids);

      const auto run_start = Clock::now();
      auto token_output_result = module.execute("token_embedding", inputs);
      const auto run_end = Clock::now();
      token_timing.run_ms = DurationMs(run_end - run_start).count();

      if (token_output_result.error() != Error::Ok) {
        std::cerr << "token_embedding execution failed: error code "
                  << static_cast<int>(token_output_result.error()) << std::endl;
        return 1;
      }

      token_executed = true;
      const auto& outputs = token_output_result.get();
      if (!outputs.empty() && outputs[0].isTensor()) {
        print_tensor_summary("token_embedding output", outputs[0].toTensor());
        token_output = outputs[0];
      }
    }

    // text_decoder
    {
      const auto [err, load_ms] = measure_method_load("text_decoder");
      if (err != Error::Ok) {
        std::cerr << "Failed to load method text_decoder: error code "
                  << static_cast<int>(err) << std::endl;
        return 1;
      }
      text_timing.load_ms = load_ms;

      std::vector<EValue> inputs;
      if (token_executed) {
        if (token_output.isTensor()) {
          inputs.emplace_back(token_output);
        }
      }

      if (inputs.empty()) {
        inputs.emplace_back(create_fallback_text_embedding());
      }

      inputs.emplace_back(create_positions_input());

      const auto run_start = Clock::now();
      Result<std::vector<EValue>> output_result =
          module.execute("text_decoder", inputs);
      const auto run_end = Clock::now();
      text_timing.run_ms = DurationMs(run_end - run_start).count();

      if (output_result.error() != Error::Ok) {
        std::cerr << "text_decoder execution failed: error code "
                  << static_cast<int>(output_result.error()) << std::endl;
        return 1;
      }

      const auto& outputs = output_result.get();
      if (!outputs.empty() && outputs[0].isTensor()) {
        print_tensor_summary("text_decoder output", outputs[0].toTensor());
      }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Program load latency (ms): " << program_load_latency.count()
              << std::endl;

    std::cout << "Method load latency (ms):" << std::endl;
    std::cout << "  audio_encoder: " << audio_timing.load_ms << std::endl;
    std::cout << "  token_embedding: " << token_timing.load_ms << std::endl;
    std::cout << "  text_decoder: " << text_timing.load_ms << std::endl;

    std::cout << "Run latency (ms):" << std::endl;
    std::cout << "  audio_encoder: " << audio_timing.run_ms << std::endl;
    std::cout << "  token_embedding: " << token_timing.run_ms << std::endl;
    std::cout << "  text_decoder: " << text_timing.run_ms << std::endl;

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Unhandled exception: " << ex.what() << std::endl;
    return 1;
  }
}
